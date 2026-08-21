# توثيق API — منصة التحليل الأكاديمي الذكي

خدمة Python/Flask واحدة (`combined_app.py`) فيها 4 نقاط اتصال (endpoints).
مش محتاج تفهم أي حاجة عن الـML أو Python — بس ابعت JSON، وارجعلك JSON.

**رابط السيرفر الأساسي:** `http://localhost:5000` (محلياً) أو الرابط اللي هيتحدد وقت النشر.

---

## 1) GET /health
فحص سريع إن السيرفر شغّال وكل الموديلات محمّلة.

**الرد:**
```json
{
  "status": "ok",
  "models_available": ["warning_year2_model.joblib", "..."],
  "advisor_knowledge_loaded": true,
  "advisor_knowledge_chars": 9685,
  "advisor_api_key_configured": true,
  "clustering_available": true
}
```

---

## 2) GET /student/{student_id}
بحث عن بيانات طالب محفوظة عندنا (اختياري — مفيدة لو عايز تعبّي فورم تلقائياً).

**مثال:** `GET /student/220072`

**الرد (200):**
```json
{
  "student_id": "220072",
  "year1": { "fall_gpa": 2.34, "spring_gpa": 2.77, "total_courses": 13, "passed_courses": 13, "failed_courses": 0, "credits": 35, "points": 89.5, "avg_grade": 2.585, "max_grade": 4.0, "min_grade": 1.0 },
  "year2": { "...": "نفس الشكل، لو موجود" },
  "year3": { "...": "نفس الشكل، لو موجود" }
}
```
**لو الكود مش موجود (404):**
```json
{ "error": "Student 999999 not found in stored records.", "student_id": "999999" }
```

---

## 3) POST /predict
النقطة الرئيسية — التنبؤات الخمسة.

**الطلب (JSON):**
```json
{
  "student_id": "220072",
  "year1": {                      // إجباري (الحد الأدنى)
    "fall_gpa": 2.34, "spring_gpa": 2.77,
    "total_courses": 13, "passed_courses": 13, "failed_courses": 0,
    "credits": 35, "points": 89.5,
    "avg_grade": 2.585, "max_grade": 4.0, "min_grade": 1.0
  },
  "year2": { "...": "اختياري، نفس الشكل" },
  "year3": { "...": "اختياري، نفس الشكل" }
}
```
كل ما بعتّ سنين أكتر، الردود بتبقى أدق (بيستخدم تلقائياً أدق موديل متاح حسب البيانات المرسلة).

**الرد:**
```json
{
  "student_id": "220072",
  "years_provided": ["year1", "year2", "year3"],
  "predictions": {
    "warning_risk": {
      "available": true,
      "probability": 0.0068,
      "predicts": "warned_in_year4",
      "target_year": 4,
      "based_on": "years1-3",
      "confidence_note": "validated (recall ~73% in testing)",
      "explanation": [
        { "factor": "معدل الفصل الأول (سنة 1)", "student_value": 2.34, "effect": "decreases_risk", "impact": 0.15 }
      ]
    },
    "not_promoted_risk": { "...": "نفس شكل warning_risk" },
    "delayed_progression_risk": { "...": "نفس الشكل" },
    "predicted_next_year_gpa": {
      "available": true,
      "value": 3.15,
      "predicts": "year4_avg_gpa",
      "target_year": 4,
      "confidence_note": "good fit (typical error ~0.17 GPA points)",
      "explanation": [ "..." ]
    },
    "graduation_probability": {
      "available": true,
      "probability": 0.953,
      "confidence_note": "LOW CONFIDENCE - trained on only 27 non-graduate examples...",
      "explanation": [ "..." ]
    },
    "peer_cluster": {
      "available": true,
      "cluster_id": 0,
      "description": "أداء أعلى من المتوسط: معدل تراكمي أعلى، إنذارات أقل بكتير...",
      "confidence_note": "تقريبي (47% من البيانات معروفة بس) - مؤشر أولي وليس تصنيفاً نهائياً"
    }
  }
}
```

### ملاحظات مهمة للباك اند
- **كل مؤشر ليه `confidence_note` نصي** — لازم يتعرض للمستخدم النهائي، مش يتخبى. بعض المؤشرات (`graduation_probability`, `delayed_progression_risk`, `peer_cluster`) أضعف من غيرها بشكل صريح ومكتوب.
- **لما `"available": false`**، فيه `"reason"` بدل النتيجة — يعني البيانات المرسلة مش كافية لهذا المؤشر بالذات (مثلاً `graduation_probability` محتاجة الثلاث سنين مع بعض).
- **الحقول المفقودة اختيارية جوه كل سنة** — لو مش عارف رقم معيّن، سيبه فاضي بدل ما تبعت صفر (الصفر بيتفسّر كقيمة حقيقية، مش "مش معروف").

---

## 4) POST /chat
المستشار الأكاديمي الذكي (محادثة نصية).

**الطلب:**
```json
{
  "gpa": 2.8,
  "student_id": "220072",        // اختياري - لو موجود، الرد بيبقى مخصص ببيانات الطالب الحقيقية
  "history": [                    // اختياري - لاستمرار محادثة سابقة
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "message": "إزاي أحسّن أدائي الأكاديمي؟"
}
```

**الرد:**
```json
{
  "reply": "بناءً على بياناتك...",
  "level": "جيد",
  "risk_context_used": true
}
```
`risk_context_used: true` معناها الرد فعلاً استخدم بيانات الطالب الحقيقية (مش نصيحة عامة).

**لو مفتاح الـAI مش متظبط على السيرفر (500):**
```json
{ "error": "OPENROUTER_API_KEY is not set. Set it as an environment variable before starting the server." }
```

---

## 5) GET /student/{student_id}/courses
تاريخ المقررات الفعلي للطالب + تنبؤات بمخاطر مقررات قادمة (بناءً على أداء المتطلب السابق فقط).

**مثال:** `GET /student/220072/courses`

**الرد (200):**
```json
{
  "student_id": "220072",
  "course_history": [
    { "course_code": "CS403", "course_name": "تعلم الالة", "academic_year": 4, "semester": "Fall", "numeric_grade": 3.0, "letter_grade": "B", "passed": true }
  ],
  "upcoming_course_risks": [
    {
      "course_code": "CS203",
      "prerequisite_course": "CS201",
      "your_prerequisite_grade": 2.0,
      "prerequisite_grade_bucket": "below C (< 2.4)",
      "historical_fail_rate": 0.0,
      "based_on_n_students": 6,
      "note": "Of 6 past students who scored below C (< 2.4) in CS201, 0% went on to fail CS203."
    }
  ],
  "coverage_note": "upcoming_course_risks only covers courses with an officially confirmed prerequisite and enough historical students..."
}
```

### ملاحظات مهمة
- **`course_history`** متاح لأي طالب عنده بيانات مقررات — تقرير مباشر من درجات حقيقية، مش تنبؤ.
- **`upcoming_course_risks`** مش موديل ML — هو **إحصاء تاريخي مباشر**: "من بين X طالب سابق بنفس درجة المتطلب، كام بالمية رسبوا في المقرر التالي؟"
- **مغطّى حالياً: 16 من أصل 53 مقرر فقط** (اللي ليها متطلب سابق مؤكد رسمياً + عينة كافية 15+ طالب). باقي مقررات CS3xx/4xx التخصصية (CS301-CS316, CS402, CS404-CS418) هتظهر في `course_history` بس، من غير `upcoming_course_risks` — لأننا مش عندنا متطلباتها الرسمية موثقة لحد دلوقتي.

---

## تشغيل السيرفر (لمين هيستضيفه)
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-..."     # أو $env:OPENROUTER_API_KEY="..." في PowerShell
python combined_app.py
```
هيشتغل على `http://0.0.0.0:5000`. لازم فولدر `ml_models/` و`student_data/` و`knowledge_base.txt` و`cluster_features.json` و`cluster_profiles.csv` يكونوا جنب `combined_app.py` بالظبط.
