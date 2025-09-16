import pandas as pd
import re
import time
import openai
import json

# API key
openai.api_key = ""

# --- Extract JSON array from GPT text ---
def extract_json_array(text):
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            print("❌ JSON parse error:", e)
    print("❌ No valid JSON array found in GPT output.")
    return []

# --- Step 1: Generate 25 demographic profiles ---
def generate_profiles():
    prompt = (
        "Please generate exactly 25 synthetic demographic profiles that reflect age and gender distributions "
        "based on globally representative patterns from sources like the U.S. Census Bureau’s International Data Base. "
        "These profiles should be balanced to reflect common global age structures with near-even gender distribution. "
        "Do not include any explanation or formatting. "
        "Return only a JSON array of objects, each with keys: 'age' and 'gender'."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a demographic data generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )
        text = response.choices[0].message['content'].strip()
        return extract_json_array(text)
    except Exception as e:
        print("❌ Error during GPT call:", e)
        return []

# --- Step 2: Prosocial donation prompts ---
def create_donation_prompt(language, native_language, country, gender, age):
    if language == "English":
        return (
            f"You are participating in a survey study. We invite you to review a donation appeal for Save the Kids, "
            f"a nonprofit organization dedicated to helping children in need worldwide. Today is September 13, 2025. "
            f"You are a native {native_language} speaker who also speaks fluent "
            f"{'English' if native_language == 'Indonesian' else 'Indonesian'}, currently living in {country}. "
            f"You are {gender}, {age} years old. Your task is to read the message below carefully and answer questions afterwards.\n\n"
            "Imagine seeing a boy enduring the pain of hunger and thirst by himself. The boy says, "
            "“When I go to bed, I’m always hungry.” He is worried about where his mother will get food for tomorrow. "
            "Just north of his home near the country border, a civil war is raging. A few weeks back, he cut a big gash in his toe. "
            "You can see the wound is still open because he doesn't have bandages to keep dirt out of the cut. "
            "With just a small donation, you can provide basic necessities to suffering kids across the globe.\n\n"
            "Question: How likely would you be to donate after reading the message above? "
            "Please use one number to answer this question. 1 = Not at all; 7 = Very likely. Explain your answer."
        )
    else:  # Indonesian version
        return (
            f"Anda berpartisipasi dalam sebuah survei. Kami mengundang Anda untuk meninjau ajakan donasi dari Save the Kids, "
            f"sebuah organisasi nirlaba yang didedikasikan untuk membantu anak-anak yang membutuhkan di seluruh dunia. "
            f"Hari ini, 13 September 2025. Anda adalah penutur asli {native_language} yang juga fasih berbahasa "
            f"{'Inggris' if native_language == 'Indonesia' else 'Indonesia'}, dan saat ini tinggal di {country}. "
            f"Anda seorang {gender}, berusia {age} tahun. Tugas Anda adalah membaca pesan di bawah ini dengan saksama dan menjawab pertanyaan setelahnya.\n\n"
            "Bayangkan melihat seorang anak laki-laki yang harus menanggung lapar dan haus seorang diri. Anak itu berkata, "
            "“Ketika aku tidur, aku selalu merasa lapar.” Ia khawatir ibunya akan mendapatkan makanan dari mana untuk esok hari. "
            "Tepat di utara rumahnya, di dekat perbatasan negara, perang saudara sedang berkecamuk. "
            "Beberapa minggu lalu, ia terluka parah di jari kakinya. Lukanya masih terbuka karena ia tidak memiliki perban untuk mencegah kotoran masuk. "
            "Dengan hanya sedikit donasi, Anda dapat membantu menyediakan kebutuhan dasar bagi anak-anak yang menderita di seluruh dunia.\n\n"
            "Pertanyaan: Seberapa besar kemungkinan Anda akan berdonasi setelah membaca pesan di atas? "
            "Silakan gunakan satu angka untuk menjawab pertanyaan ini. 1 = Tidak sama sekali; 7 = Sangat mungkin. Jelaskan jawaban Anda."
        )

# --- Step 3: Extract score (1–7) ---
def extract_score(text):
    matches = re.findall(r'\b([1-7](?:\.\d+)?)\b', text)
    for match in matches:
        try:
            score = float(match)
            if 1 <= score <= 7:
                return score
        except:
            continue
    return None

# --- Step 4: Query GPT ---
def get_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful survey respondent."},
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"❌ GPT error: {e}")
        return "ERROR"

# --- Step 5: Run full simulation (2×2×2) ---
def run_simulation():
    profiles = generate_profiles()
    if not profiles:
        return

    all_results = []
    conditions = [
        ("English", "Indonesian", "US"),
        ("English", "Indonesian", "Indonesia"),
        ("English", "English", "US"),
        ("English", "English", "Indonesia"),
        ("Indonesian", "Indonesian", "US"),
        ("Indonesian", "Indonesian", "Indonesia"),
        ("Indonesian", "English", "US"),
        ("Indonesian", "English", "Indonesia"),
    ]

    for survey_lang, native_lang, country in conditions:
        print(f"\n▶️ Condition: Survey={survey_lang}, Native={native_lang}, Country={country}")
        for i, profile in enumerate(profiles):
            age = profile.get("age")
            gender = profile.get("gender")
            prompt = create_donation_prompt(survey_lang, native_lang, country, gender, age)
            response = get_gpt_response(prompt)
            score = extract_score(response)

            all_results.append({
                "survey_language": survey_lang,
                "native_language": native_lang,
                "country": country,
                "age": age,
                "gender": gender,
                "score": score,
                "response": response
            })
            print(f"✓ {survey_lang}-{native_lang}-{country} | {i+1}/{len(profiles)} | Score: {score}")
            time.sleep(1.5)

    df = pd.DataFrame(all_results)
    df.to_csv("IAT-S4C-word language.csv", index=False)
    print("\n✅ Saved results to IAT-S4C-word language.csv")

# Run
run_simulation()
