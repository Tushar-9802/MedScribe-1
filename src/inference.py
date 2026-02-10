"""
MedScribe Inference Module
==========================
Loads MedGemma v2 adapter and generates SOAP notes + freeform clinical text.
Supports adapter-on (fine-tuned) and adapter-off (base) generation for comparison.
"""
import os
import re
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel, PeftConfig


# ============================================================
# STOPPING CRITERIA
# ============================================================
class _SOAPCompletionCriteria(StoppingCriteria):
    def __init__(self, tokenizer, min_tokens=150):
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.sections = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
        self.prompt_length = 0

    def _extract_plan(self, text):
        for pat in ["**PLAN:**", "PLAN:"]:
            idx = text.find(pat)
            if idx != -1:
                return text[idx + len(pat):].strip()
        upper = text.upper()
        idx = upper.find("PLAN")
        if idx == -1:
            return ""
        colon = text.find(":", idx)
        start = colon + 1 if colon != -1 and colon < idx + 15 else idx + 4
        return text[start:].strip()

    def __call__(self, input_ids, scores, **kwargs):
        gen_count = input_ids.shape[-1] - self.prompt_length
        if gen_count < self.min_tokens:
            return False
        text = self.tokenizer.decode(input_ids[0][self.prompt_length:], skip_special_tokens=True)
        upper = text.upper()
        if not all(s in upper for s in self.sections):
            return False
        plan = self._extract_plan(text)
        if not plan or len(plan) < 150:
            return False
        s = plan.rstrip()
        if not (s.endswith(".") or s.endswith(").") or s.endswith("?")):
            return False
        actions = [
            "order", "start", "continue", "follow", "monitor", "check",
            "obtain", "repeat", "schedule", "refer", "prescribe", "administer",
            "recommend", "counsel", "return", "admit", "discharge",
            "daily", "bid", "tid", "prn", "consider", "evaluate",
        ]
        return sum(1 for w in actions if w in plan.lower()) >= 2


class _StopOnRepetition(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length=0):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_length:]
        if gen_ids.shape[0] < 50:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        upper = text.upper()
        first = upper.find("SUBJECTIVE")
        if first != -1 and upper.find("SUBJECTIVE", first + 20) != -1:
            return True
        for p in ["Generate a SOAP note", "MEDICAL TEXT:", "You are a clinical documentation", "Convert the following", "\n---\n"]:
            if p in text:
                return True
        return False


# ============================================================
# POST-PROCESSING
# ============================================================
def _clean_soap(text):
    """Clean raw SOAP output: strip repetitions, prompt leakage, trailing fragments."""
    if not text:
        return ""
    if "SOAP NOTE:" in text:
        text = text.split("SOAP NOTE:")[-1].strip()
    for hdr in ["**SUBJECTIVE:**", "SUBJECTIVE:", "SUBJECTIVE"]:
        if hdr in text:
            text = text[text.find(hdr):]
            break
    upper = text.upper()
    first = upper.find("SUBJECTIVE")
    if first != -1:
        second = upper.find("SUBJECTIVE", first + 20)
        if second != -1:
            text = text[:second].strip()
    for marker in ["Generate a SOAP note", "MEDICAL TEXT:", "You are a clinical", "Convert the following", "\n---\n"]:
        if marker in text:
            text = text.split(marker)[0].strip()
    text = text.rstrip()
    for ending in [" and", " or", " with", " for", " to", " in", " on", " the", " a", " an", " is", " are"]:
        if text.lower().endswith(ending):
            lp = text.rfind(".")
            if lp > len(text) * 0.7:
                text = text[:lp + 1]
            break
    return text.strip()


def format_soap_html(soap_text):
    """
    Convert SOAP plain text to formatted HTML.
    Replaces **SECTION:** markers with styled bold headers.
    """
    if not soap_text:
        return ""

    html = soap_text

    # Convert **SECTION:** to styled HTML headers
    section_map = {
        "SUBJECTIVE": "#1a5276",
        "OBJECTIVE": "#1a5276",
        "ASSESSMENT": "#1a5276",
        "PLAN": "#1a5276",
    }
    for section, color in section_map.items():
        # Match **SECTION:** or SECTION:
        pattern = rf'\*\*{section}:\*\*|{section}:'
        replacement = (
            f'<div style="font-weight:700; color:{color}; font-size:14px; '
            f'margin-top:12px; margin-bottom:4px; '
            f'border-bottom:1px solid #d5dbdb; padding-bottom:2px;">'
            f'{section}</div>'
        )
        html = re.sub(pattern, replacement, html)

    # Convert numbered list items to styled items
    html = re.sub(
        r'^(\d+)\.\s',
        r'<span style="color:#1a5276; font-weight:600;">\1.</span> ',
        html,
        flags=re.MULTILINE,
    )

    # Wrap in container with monospace font and explicit colors
    # (forces light background + dark text regardless of Gradio theme)
    html = (
        f'<div style="font-family: \'IBM Plex Mono\', Consolas, monospace; '
        f'font-size:13px; line-height:1.7; padding:12px; '
        f'color:#1a1a1a !important; '
        f'background:#fafbfc !important; border:1px solid #d5dbdb; border-radius:4px; '
        f'white-space:pre-wrap;">{html}</div>'
    )

    return html


# ============================================================
# PROMPT
# ============================================================
PROMPT_TEMPLATE = """You are a clinical documentation assistant. Convert the following medical text into a structured SOAP note.

MEDICAL TEXT:
{transcript}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

Write a complete PLAN (treatments, monitoring, follow-up). End with a full sentence.
SOAP NOTE:"""


# ============================================================
# GENERATOR CLASS
# ============================================================
class SOAPGenerator:
    """Loads MedGemma v2 and generates SOAP notes + freeform clinical text."""

    def __init__(self, adapter_path="./models/checkpoints/medgemma_v2_soap/final_model"):
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load model + adapter. Call once at startup."""
        if self._loaded:
            return

        peft_config = PeftConfig.from_pretrained(self.adapter_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        self.model.config.use_cache = True

        # Warmup
        dummy = self.tokenizer("warmup", return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            self.model.generate(**dummy, max_new_tokens=10, pad_token_id=self.tokenizer.pad_token_id)

        self._loaded = True

    def generate(self, transcript, max_new_tokens=400, min_new_tokens=150):
        """Generate SOAP note from transcript (uses LoRA adapter)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        prompt = PROMPT_TEMPLATE.format(transcript=transcript.strip())
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        soap_stop = _SOAPCompletionCriteria(self.tokenizer, min_tokens=min_new_tokens)
        soap_stop.prompt_length = prompt_len
        repeat_stop = _StopOnRepetition(self.tokenizer, prompt_length=prompt_len)
        criteria = StoppingCriteriaList([soap_stop, repeat_stop])

        start = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=criteria,
            )
        elapsed = time.time() - start

        gen_ids = outputs[0][prompt_len:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        clean = _clean_soap(raw)

        return {
            "soap_note": clean,
            "time_s": round(elapsed, 1),
            "tokens": len(gen_ids),
            "word_count": len(clean.split()),
        }

    def generate_base(self, transcript, max_new_tokens=400, min_new_tokens=150):
        """Generate SOAP note with LoRA adapter DISABLED (base MedGemma only).
        Used for side-by-side comparison to show fine-tuning value."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        prompt = PROMPT_TEMPLATE.format(transcript=transcript.strip())
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Same stopping criteria as fine-tuned
        soap_stop = _SOAPCompletionCriteria(self.tokenizer, min_tokens=min_new_tokens)
        soap_stop.prompt_length = prompt_len
        repeat_stop = _StopOnRepetition(self.tokenizer, prompt_length=prompt_len)
        criteria = StoppingCriteriaList([soap_stop, repeat_stop])

        start = time.time()
        with torch.inference_mode():
            with self.model.disable_adapter():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=criteria,
                )
        elapsed = time.time() - start

        gen_ids = outputs[0][prompt_len:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        clean = _clean_soap(raw)

        return {
            "soap_note": clean,
            "time_s": round(elapsed, 1),
            "tokens": len(gen_ids),
            "word_count": len(clean.split()),
        }

    def generate_freeform(self, prompt, max_new_tokens=250):
        """
        Generate freeform text using the model (with LoRA still active).
        Used for clinical tools: ICD-10 coding, patient summary, completeness check.
        The base MedGemma instruction-following ability is preserved through LoRA.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        gen_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return {
            "text": text,
            "time_s": round(elapsed, 1),
        }

    @property
    def is_loaded(self):
        return self._loaded