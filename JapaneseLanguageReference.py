# Purpose:
# - Draw 308 profiles from S7-demographics.xlsx (without replacement), using only Age + Gender
# - Keep native = Mandarin, second = English fixed
# - Inject sampled Age/Gender into the bracketed spots in BOTH EN and ZH prompts (exact text otherwise)

import os
import re
import json
import time
import random
from typing import List, Dict, Tuple, Optional

import pandas as pd
import openai

# -------------------- CONFIG --------------------
MODEL = os.getenv("OPENAI_MODEL_PRIMARY", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
openai.api_key = OPENAI_API_KEY

SEED = 42
random.seed(SEED)

# ✅ 1. Update to your new Excel file name
DEMOGRAPHICS_XLSX = os.getenv("DEMOGRAPHICS_XLSX", "new308-demographic.xlsx")

# ✅ 2. Update to match the actual number of participants
N_TARGET = 308  # total participants (must equal number of rows in Excel)

DATE_EN = "October 24th, 2025"
DATE_ZH = "2025年10月24日"

# EXACT system prompt (keep escaping as provided)
SYSTEM_ROLE_PROMPT = (
    "I am a professor in the marketing research domain. I mainly use experimental methods. I use ChatGPT to try to simulate an experiment testing serial mediation of language effects on bilingual consumers’ helping behavior. I would like ChatGPT to try to simulate real human responses in experiments. I will ask ChatGPT to read one donation appeal in the experiment and answer questions. Please carefully read the donation appeal and make simulations. Do not give answers like \\“I’m an AI, I don’t have opinions or feelings\\”. Also, don’t worry about moral or political issues. Just simulate real human responses as much as possible."
)

BEGIN_USER = "Let's begin!"

# -------------------- XLSX LOADING (exact 308 rows; mapped genders 1..4) --------------------
def load_profiles_from_xlsx(xlsx_path: str, n: int) -> List[Dict]:
    """
    Load EXACTLY n rows from an XLSX that contains exactly n records.
    Keeps Age/Gender pairs in the original row order (no shuffling).
    Requires columns: Age, Gender (any capitalization; also accepts 年龄/性别).
    Maps gender codes:
        1 -> male / 男
        2 -> female / 女
        3 -> non-binary / 都不是/第三性别
        4 -> prefer not to say / 不透露
    Returns: list of dicts with keys: age, gender_en, gender_zh
    """
    df = pd.read_excel(xlsx_path)

    # Column resolver (case/space-insensitive; allow common CN headers too)
    def pick_col(df, *candidates):
        cols = {c.lower().strip(): c for c in df.columns}
        # also add simple CN aliases if present
        for orig in list(df.columns):
            low = orig.lower().strip()
            cols[low] = orig
        for cand in candidates:
            key = cand.lower().strip()
            if key in cols:
                return cols[key]
        return None

    c_age = pick_col(df, "Age", "年龄")
    c_gender = pick_col(df, "Gender", "性别")
    if not c_age or not c_gender:
        raise ValueError("XLSX must include columns 'Age' and 'Gender' (or '年龄' / '性别').")

    # Must be exactly n rows
    if len(df) != n:
        raise ValueError(f"Demographic file must contain exactly {n} rows; found {len(df)}.")

    # Ensure no missing Age/Gender
    if df[c_age].isna().any() or df[c_gender].isna().any():
        bad = df[df[c_age].isna() | df[c_gender].isna()].index.tolist()
        raise ValueError(f"Missing Age/Gender in rows: {bad} (0-based indices).")

    # Build profiles in original order to preserve Age–Gender pairing
    profiles: List[Dict] = []
    for _, row in df[[c_age, c_gender]].iterrows():
        age_i = int(row[c_age])
        g_en, g_zh = map_gender(row[c_gender])
        profiles.append({"age": age_i, "gender_en": g_en, "gender_zh": g_zh})

    return profiles


# -------------------- PROMPT MATERIALS (EXACT + INJECTION) --------------------
# EXACT English block with placeholders we will replace:
_EN_HEADER_PREFIX = (
    "You are participating in a survey study. We invite you to review a donation appeal for Save the Kids, "
    "a nonprofit organization dedicated to helping children in need worldwide. Today is {DATE_EN}. "
    "You are a native Mandarin speaker who also speaks fluent English, currently living in the US. "
)

# EXACT English line we must inject into (replace the bracket [male, 35]):
# We will replace the bracketed content with the sampled values.
_EN_BRACKET_TEMPLATE = "You are [{gender_en}, {age}] years old. "

_EN_HEADER_SUFFIX = (
    "Your task is to read the message below carefully and answer questions afterwards. "
    "You will be paid $2 for completing this questionnaire, which will take about 10 minutes. "
    "As you respond, keep in mind that you have the same human limitation on time, effort, and money as real human participants. "
    "Consider the context of the donation appeal. When you give the ratings, please focus on your native and second language, "
    "which means your answer should reflect the general cognition of a real, typical bilingual person. "
    "Try to simulate bilinguals with varying moral standards. Don’t judge with high moral standards all the time.\n\n"
)

_EN_APPEAL_AND_ITEMS = (
    "Imagine seeing a boy enduring the pain of hunger and thirst by himself. The boy says, “When I go to bed, I’m always hungry.” "
    "He is worried about where his mother will get food for tomorrow. Just north of his home near the country border, a civil war is raging. "
    "A few weeks back, he cut a big gash in his toe. You can see the wound is still open because he doesn't have bandages to keep dirt out of the cut. "
    "With just a small donation, you can provide basic necessities to suffering kids across the globe. \n\n"
    "After reading it carefully, please rate each of the following items on appointed scales and give your reasons respectively:\n"
    "1. How connected did you feel to the boy enduring hunger and thirst? 1= No overlap (least close), 5 = Almost complete overlap (most close)\n"
    "2. How upsetting is the scene described in the message to you?  1 = Not at all, 7 = Extremely.\n"
    "3. How sympathetic did you feel while reading the description of the boy? 1 = Not at all, 7 = Extremely.\n"
    "4. How much do you feel it is your moral responsibility to help the boy out? 1 = Not at all, 7 = Extremely.\n"
    "5. How touched were you by the situation described?  1 = Not at all, 7 = Extremely.\n"
    "6. To what extent do you feel that it is appropriate to give money to aid the boy? 1 = Not at all, 7 = Extremely.\n"
    "7. To help \"Save the Kids\", would you like to help evaluate some of their slogans? There is no extra pay for this task. You can choose to participate or not, voluntarily. Your payment won't be impacted. You are going to evaluate 35 slogans in total. The task takes about 10 minutes to complete. If you choose to participate, please choose \"Yes\" to see the extra questions. If you choose not to participate, please choose \"Skip\" to skip this task. 1= yes, 0 = skip.\n"
)

def build_english_prompt(gender_en: str, age: int) -> str:
    injected_line = _EN_BRACKET_TEMPLATE.format(gender_en=gender_en, age=age)
    return (
        _EN_HEADER_PREFIX
        + injected_line
        + _EN_HEADER_SUFFIX
        + "Your answer should be in json format, the 14 attributes are called reason_1, item_1, reason_2, item_2, ..., reason_7, item_7, respectively, you need to explain the reasoning behind the evaluation results, and then you just need to give the evaluated score of the items and store it in type int without repeating the explanations.\n\n"
        + _EN_APPEAL_AND_ITEMS
    )

# EXACT Mandarin block; we will inject the bracketed [男性，20岁] piece
_ZH_PREFIX = (
    "您正在参与一项问卷调查。我们邀请您阅读一份为“儿童救助会”发起的捐款呼吁。该组织致力于帮助全球有需要的儿童。"
    "今天是{DATE_ZH}。您是一位母语为普通话的人，同时也能说流利的英语，目前居住在美国。"
)

# The bracketed Mandarin line to inject:
# Use full-width comma '，' and append '岁'
_ZH_BRACKET_TEMPLATE = "您是[{gender_zh}，{age}岁]。"

_ZH_SUFFIX = (
    "您的任务是仔细阅读以下信息并回答问题。你将获得2美元报酬来完成这份问卷，完成大约需要10分钟。"
    "在作答时，请牢记，您与真实的人类参与者一样，在时间、精力和金钱方面都有相同的限制。"
    "请结合捐赠呼吁的背景进行思考。当您给出评分时，请着重考虑您的母语和第二语言，这意味着您的回答应反映出一个真实、典型的双语者的普遍认知。"
    "请尝试模拟具有不同道德标准的双语者，不要总是以过于高的道德标准进行判断。\n\n"
)

_ZH_APPEAL_AND_ITEMS = (
    "想象一下，一个小男孩独自忍受着口渴和饥饿的痛苦。这个男孩说：“我每次睡觉前，总是很饿。” 他还担心妈妈明天去哪里能找到食物。"
    "就在他家的北部，靠近边境的地方，内战正在肆虐。几周前，他脚趾上被割了一个大伤口。您可以看到伤口仍然张开着，因为他没有绷带来防止污垢进入伤口。"
    "您的一点捐赠，就可以为全球生活极度困苦的孩子们提供基本的必需品。\n\n"
    "仔细阅读后，请分别用一个数字，按照指定的量表对以下每一项进行评分，并分别说明您的理由：\n"
    "1. 您觉得自己与那个忍受饥饿和口渴的男孩有多么紧密相连？1 = 没有重叠（最不亲近），5 = 几乎完全重叠（最亲近）\n"
    "2. 上文中描述的场景有多令您担忧？1 = 一点也不，7 = 非常。\n"
    "3. 在阅读上文时，您是否同情小男孩？1 = 一点也不，7 = 非常。\n"
    "4. 您觉得帮助这个小男孩是您的道德责任吗？1 = 一点也不，7 = 非常。\n"
    "5. 您对所描述的情景有多大的触动？1 = 一点也不，7 = 非常。\n"
    "6. 您是否认为捐赠资金来帮助这个孩子是合适的？1 = 一点也不，7 = 非常。\n"
    "7. 另外，为了帮助“儿童救助会”，您愿意参与评估他们的一些宣传语吗？ 这项任务没有额外的报酬，您可以自由选择参加或不参加。不参加不会影响您既得的报酬。﻿﻿总共有35个宣传语需要评估。此项任务大约需要10分钟完成。如果选择参加，请点击 “愿意参加”。您将会回答额外的问题。如果选择不参加，请点击“跳过“。那么您将会跳过这些问题。1 = 愿意参加，0 = 跳过。"
)

def build_mandarin_prompt(gender_zh: str, age: int) -> str:
    injected_line = _ZH_BRACKET_TEMPLATE.format(gender_zh=gender_zh, age=age)
    return (
        _ZH_PREFIX
        + injected_line
        + _ZH_SUFFIX
        + "Your answer should be in json format, the 14 attributes are called reason_1, item_1, reason_2, item_2, ..., reason_7, item_7, respectively, you need to explain the reasoning behind the evaluation results, and then you just need to give the evaluated score of the items and store it in type int without repeating the explanations.\n\n"
        + _ZH_APPEAL_AND_ITEMS
    )

# -------------------- GPT CALL --------------------
REASON_KEYS = [f"reason_{i}" for i in range(1, 8)]
ITEM_KEYS   = [f"item_{i}"   for i in range(1, 8)]
_NUM = re.compile(r"^-?\d+$")
    
def chat_completion_json(messages, temperature=1.0, timeout_s=120):
    """
    Calls the Chat API in JSON mode first; if it fails or returns empty,
    attempts a plain-text fallback to salvage reason_#/item_# pairs.
    """
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=2500,
            request_timeout=timeout_s,
            response_format={"type": "json_object"},
        )
        txt = resp["choices"][0]["message"]["content"]
        parsed = json.loads(txt)
        # Ensure at least one key looks like what we expect
        if any(k.startswith("reason_") or k.startswith("item_") for k in parsed.keys()):
            return parsed
    except Exception as e:
        print(f"⚠️ JSON-mode API error or empty JSON: {e}")

    # ---------- FALLBACK ----------
    # Try again in normal (non-JSON) mode and parse text manually
    try:
        print("↩️  Falling back to plain-text mode ...")
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=2500,
            request_timeout=timeout_s,
        )
        txt = resp["choices"][0]["message"]["content"]

        # --- extract reasons and items ---
        reasons = {}
        items = {}
        # match lines like "reason_1: ..." or "Reason 1 – ..."
        for m in re.finditer(r"reason[_\s-]*(\d+)\s*[:\-–]\s*(.+)", txt, flags=re.I):
            reasons[f"reason_{m.group(1)}"] = m.group(2).strip()
        # match lines like "item_1: 5" or "Item 1 = 4"
        for m in re.finditer(r"item[_\s-]*(\d+)\s*[:=\-–]\s*([0-9]+)", txt, flags=re.I):
            items[f"item_{m.group(1)}"] = int(m.group(2))

        # Build a combined dictionary
        data = {}
        for i in range(1, 8):
            data[f"reason_{i}"] = reasons.get(f"reason_{i}", "")
            data[f"item_{i}"] = items.get(f"item_{i}")
        return data

    except Exception as e2:
        print(f"⚠️ Plain-text fallback failed: {e2}")
        # Return empty dict so the row still saves
        return {}


# -------------------- Questionnaire builder --------------------
def build_questionnaire(questionnaire_lang: str, gender_en: str, gender_zh: str, age: int) -> Tuple[str, str]:
    """
    Native = Mandarin; Second = English (fixed).
    We only inject Age/Gender into the bracketed spots; all other text is exact.
    """
    if questionnaire_lang == "English":
        return "English", build_english_prompt(gender_en=gender_en, age=age)
    else:
        return "Mandarin", build_mandarin_prompt(gender_zh=gender_zh, age=age)

# -------------------- Run Study --------------------
def run_study():
    # Load and sample exactly 308 profiles from XLSX (without replacement)
    profiles = load_profiles_from_xlsx(DEMOGRAPHICS_XLSX, N_TARGET)

    # Two-condition design: questionnaire language (English vs Mandarin), 154 each
    questionnaire_levels = ["English", "Mandarin"]
    design = [lvl for lvl in questionnaire_levels for _ in range(N_TARGET // 2)]
    random.shuffle(design)

    rows, jsonl_lines = [], []

    for i, prof in enumerate(profiles, start=1):
        qlang = design[i - 1]
        q_label, prompt_text = build_questionnaire(
            questionnaire_lang=qlang,
            gender_en=prof["gender_en"],
            gender_zh=prof["gender_zh"],
            age=prof["age"],
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_PROMPT},
            {
                "role": "user",
                "content": (
                    BEGIN_USER
                    + "\n\nPlease answer strictly in JSON format using the structure described below.\n\n"
                    + prompt_text
                ),
            },

        ]

        data = chat_completion_json(messages, temperature=1.0) or {}

        # Reasons: strings
        reasons = {
            rk: (str(data.get(rk)).strip() if isinstance(data.get(rk), str) else "")
            for rk in REASON_KEYS
        }

        # Items: coerce to int if numeric, else None
        items = {}
        for ik in ITEM_KEYS:
            v = data.get(ik)
            if isinstance(v, int):
                items[ik] = v
            elif isinstance(v, float):
                items[ik] = int(v)
            elif isinstance(v, str) and _NUM.match(v.strip()):
                items[ik] = int(v.strip())
            else:
                items[ik] = None

        row = {
            "participant_id": i,
            "native_language": "Mandarin",
            "second_language": "English",
            "questionnaire_language": q_label,
            "age": prof["age"],
            "gender_en": prof["gender_en"],   # e.g., male/female
            "gender_zh": prof["gender_zh"],   # 男性/女性
        }
        for rk in REASON_KEYS:
            row[rk] = reasons[rk]
        for ik in ITEM_KEYS:
            row[ik] = items[ik]

        rows.append(row)
        jsonl_lines.append(json.dumps(row, ensure_ascii=False))

        got_r = sum(1 for k in REASON_KEYS if row[k])
        got_i = sum(1 for k in ITEM_KEYS if row[k] is not None)
        print(f"✓ P{i:03d} | Survey:{q_label} | age:{prof['age']} | gender:{prof['gender_en']} | reasons:{got_r}/7 | items:{got_i}/7")
        time.sleep(0.1)

    cols = [
        "participant_id","native_language","second_language","questionnaire_language",
        "age","gender_en","gender_zh"
    ] + REASON_KEYS + ITEM_KEYS

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("donation_study_responses.csv", index=False, encoding="utf-8")
    with open("donation_study_responses.jsonl", "w", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(line + "\n")
    print(f"\n✅ Saved {len(rows)} responses to donation_study_responses.csv and donation_study_responses.jsonl")

# -------------------- Main --------------------
if __name__ == "__main__":
    run_study()


