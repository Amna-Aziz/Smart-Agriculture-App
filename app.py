import streamlit as st
import numpy as np
import pickle
import requests
import re
from io import BytesIO
from PIL import Image, ImageFilter
from datetime import datetime
from gtts import gTTS
from keras.applications.resnet50 import preprocess_input
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AgriSmart AI - Decision Support System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

URDU_UI = {
    "Language / زبان": "زبان منتخب کریں",
    "🌱 AgriSmart AI Decision Support System": "🌱 ایگری اسمارٹ اے آئی فیصلہ معاون نظام",
    "Plant Pathology Diagnosis · Soil Chemistry Advisory · Regional Weather Insights": "پودوں کی بیماریوں کی تشخیص · مٹی کی غذائیت · علاقائی موسم",
    "🎯 97.36% Validation Accuracy": "🎯 توثیق کی درستگی 97.36٪",
    "🧬 ResNet50 Fine-Tuned": "🧬 ResNet50 کی تربیت شدہ قسم",
    "🌾 19 Disease Classes": "🌾 بیماری کی 19 اقسام",
    "📊 Session Summary": "📊 سیشن کا خلاصہ",
    "Diagnoses Run": "تشخیصات کی تعداد",
    "Model loaded": "ماڈل لوڈ ہو گیا",
    "classes": "اقسام",
    "Model failed to load": "ماڈل لوڈ نہیں ہو سکا",
    "🕘 Recent History": "🕘 حالیہ تشخیصات",
    "No diagnoses yet this session.": "اس سیشن میں ابھی کوئی تشخیص نہیں ہوئی۔",
    "{time} — **{disease}** ({confidence:.1f}%)": "{time} — **{disease}** ({confidence:.1f}٪)",
    "🗑️ Clear History": "🗑️ تاریخ صاف کریں",
    "🍃 Leaf Disease Diagnosis": "🍃 پتوں کی بیماری کی تشخیص",
    "🧪 Soil Advisory": "🧪 مٹی کا مشورہ",
    "🌤️ Weather Engine": "🌤️ موسمی معلومات",
    "Plant Leaf Pathogen Identification": "پودوں کے پتوں کی بیماری کی شناخت",
    "Upload a leaf image to run AI-powered disease classification and get an actionable treatment plan.": "بیماری کی شناخت اور علاج کا مشورہ حاصل کرنے کے لیے پتے کی تصویر اپ لوڈ کریں۔",
    "Upload a leaf image on the left to see diagnostic results here.": "تشخیص کے نتائج دیکھنے کے لیے بائیں جانب پتے کی تصویر اپ لوڈ کریں۔",
    "Upload Leaf Image": "پتے کی تصویر اپ لوڈ کریں",
    "Uploaded Leaf Image": "اپ لوڈ کردہ پتے کی تصویر",
    "🔍 Show Disease Severity Overlay": "🔍 بیماری کی شدت کا اوورلے دکھائیں",
    "Estimated Disease Severity Overlay": "بیماری کی شدت کا تخمینی اوورلے",
    "Red overlay marks likely discolored spots; it is a visual estimate, not confirmed disease segmentation.": "سرخ نشان ممکنہ بے رنگ دھبوں کا بصری اندازہ ہیں، بیماری سے متاثرہ حصے کی تصدیق شدہ نشاندہی نہیں۔",
    "No obvious discolored spots were highlighted; this does not rule out disease.": "رنگ بدلے ہوئے واضح دھبے نمایاں نہیں ہوئے؛ اس سے بیماری کی موجودگی رد نہیں ہوتی۔",
    "Model could not be loaded. Check that resnet_plant_disease_model.pkl is in the app folder.": "ماڈل لوڈ نہیں ہو سکا۔ تصدیق کریں کہ resnet_plant_disease_model.pkl ایپ کے فولڈر میں موجود ہے۔",
    "Analyzing leaf image...": "پتے کی تصویر کا تجزیہ ہو رہا ہے...",
    "📊 Diagnostic Results": "📊 تشخیص کے نتائج",
    "Identified Condition": "شناخت شدہ بیماری",
    "Detected Condition": "شناخت شدہ بیماری",
    "Confidence": "اعتماد",
    "Model Confidence": "ماڈل کا اعتماد",
    "Estimated Affected Area (%)": "متاثرہ رقبے کا تخمینہ (%)",
    "Spread Risk Level": "بیماری پھیلنے کا خطرہ",
    "Urgency of Treatment": "علاج کی فوری ضرورت",
    "Severity-based estimate": "شدت کی بنیاد پر تخمینہ",
    "Visual Breakdown": "خطرے کی تفصیل",
    "Low": "کم",
    "Moderate": "درمیانہ",
    "High": "زیادہ",
    "Routine": "معمول کے مطابق",
    "Prompt": "جلد",
    "Immediate": "فوری",
    "Low Confidence Detection: The uploaded leaf image may belong to an unsupported crop or symptoms are unclear. Please upload a clear, close-up photo of an affected leaf.": "کم اعتماد کے ساتھ تشخیص: اپ لوڈ کی گئی تصویر کسی غیر معاون فصل کی ہو سکتی ہے یا علامات واضح نہیں ہیں۔ براہِ کرم متاثرہ پتے کی صاف اور قریب سے لی گئی تصویر اپ لوڈ کریں۔",
    "Severity": "شدت",
    "✅ Healthy": "✅ صحت مند",
    "🔴 High Risk": "🔴 زیادہ خطرہ",
    "🟠 Moderate Risk": "🟠 درمیانہ خطرہ",
    "📋 Recommended Action Plan": "📋 تجویز کردہ اقدامات",
    "⬇️️ Download Diagnosis Report (.txt)": "⬇️️ تشخیصی رپورٹ ڈاؤن لوڈ کریں (.txt)",
    "⬇️ Download PDF Diagnosis Report": "⬇️ پی ڈی ایف تشخیصی رپورٹ ڈاؤن لوڈ کریں",
    "🔊 Generate audio report": "🔊 آڈیو رپورٹ بنائیں",
    "Generating audio report...": "آڈیو رپورٹ بنائی جا رہی ہے...",
    "Could not generate audio. Check your internet connection and try again.": "آڈیو نہیں بن سکی۔ انٹرنیٹ کنکشن چیک کرکے دوبارہ کوشش کریں۔",
    "⚠️ High-risk disease detected — review treatment plan below": "⚠️ زیادہ خطرے والی بیماری ملی ہے؛ نیچے دیا گیا علاج دیکھیں۔",
    "✅ Diagnosis complete": "✅ تشخیص مکمل ہو گئی۔",
    "#### 🤖 Ensemble Intelligence (Automated HF API Cross-Check)": "#### 🤖 دو ماڈلز کا خودکار موازنہ",
    "HF_API_TOKEN is missing. Add a Hugging Face token with Inference Providers permission to `.streamlit/secrets.toml`.": "HF_API_TOKEN موجود نہیں۔ Inference Providers اجازت والا Hugging Face ٹوکن `.streamlit/secrets.toml` میں شامل کریں۔",
    "Hugging Face model is loading. Please retry shortly.": "Hugging Face ماڈل لوڈ ہو رہا ہے۔ کچھ دیر بعد دوبارہ کوشش کریں۔",
    "Hugging Face returned an unexpected response. The primary ResNet50 prediction remains active.": "Hugging Face سے غیر متوقع جواب ملا۔ بنیادی ResNet50 نتیجہ برقرار ہے۔",
    "🤖 Agri Assistant": "🤖 زرعی معاون",
    "Ask about crop diseases, spray practices, soil nutrients, or field conditions.": "فصل کی بیماریوں، اسپرے، مٹی کی غذائیت یا کھیت کے حالات کے بارے میں پوچھیں۔",
    "Hi! Tell me your crop and ask about symptoms, sprays, soil health, or weather.": "سلام! اپنی فصل بتائیں اور علامات، اسپرے، مٹی یا موسم کے بارے میں پوچھیں۔",
    "Ask the Agri Assistant...": "زرعی معاون سے سوال کریں...",
    "Soil Chemistry & Dosing Strategy": "مٹی کی کیمیائی حالت اور کھاد کی حکمتِ عملی",
    "Enter your soil lab test values to receive a customized nutrient advisory.": "غذائی اجزا کا مشورہ حاصل کرنے کے لیے مٹی کے لیبارٹری نتائج درج کریں۔",
    "Nitrogen (N) - mg/kg": "نائٹروجن (N) - mg/kg",
    "Phosphorus (P) - mg/kg": "فاسفورس (P) - mg/kg",
    "Potassium (K) - mg/kg": "پوٹاشیم (K) - mg/kg",
    "Soil pH Level": "مٹی کی pH سطح",
    "Compute Soil Nutrient Advisory": "مٹی کے غذائی اجزا کا مشورہ حاصل کریں",
    "### 🧪 Soil Health Assessment": "### 🧪 مٹی کی صحت کا جائزہ",
    "NPK Fertilizer Requirement Calculator": "NPK کھاد کی ضرورت کا کیلکولیٹر",
    "Enter crop-specific nutrient targets. Rates are kg/acre; soil-test values above are mg/kg and are not converted directly. ": "فصل کے مطابق غذائی ہدف درج کریں۔ شرح kg/acre میں ہے؛ اوپر کے مٹی ٹیسٹ mg/kg میں ہیں اور براہِ راست تبدیل نہیں ہوتے۔ ",
    "Marlas are converted at 160 marlas per acre. Calculations assume Urea 46-0-0, DAP 18-46-0, MOP 0-0-60, and 50 kg per bag.": "حساب میں 160 مرلے = 1 ایکڑ، یوریا 46-0-0، DAP 18-46-0، MOP 0-0-60 اور فی بوری 50 کلوگرام فرض کیے گئے ہیں۔",
    "Field area unit": "رقبے کی اکائی",
    "Acres": "ایکڑ",
    "Marlas": "مرلے",
    "Target Nitrogen (N) - kg/acre": "نائٹروجن کا ہدف (N) - kg/acre",
    "Target Phosphate (P₂O₅) - kg/acre": "فاسفیٹ کا ہدف (P₂O₅) - kg/acre",
    "Target Potash (K₂O) - kg/acre": "پوٹاش کا ہدف (K₂O) - kg/acre",
    "#### Estimated Fertilizer Requirement": "#### کھاد کی اندازاً ضرورت",
    "Urea": "یوریا",
    "DAP": "DAP",
    "Potash (MOP)": "پوٹاش (MOP)",
    "full bag": "پوری بوری",
    "full bags": "پوریاں بوریاں",
    "to cover requirement": "ضرورت پوری کرنے کے لیے",
    "Regional Weather Advisory": "علاقائی موسمی مشورہ",
    "Live weather data powered by OpenWeatherMap.": "موسم کی تازہ معلومات۔",
    "Enter City / Region": "شہر / علاقہ درج کریں",
    "Fetch Advisory": "موسمی مشورہ حاصل کریں",
    "Weather API key not configured. Add OPENWEATHER_API_KEY in Streamlit secrets.": "Weather API کی موجود نہیں۔ Streamlit secrets میں OPENWEATHER_API_KEY شامل کریں۔",
    "• **Nitrogen Deficiency:** Apply Urea (46% N) at the recommended dosage to boost vegetative growth.": "• **نائٹروجن کی کمی:** پودوں کی بڑھوتری کے لیے تجویز کردہ مقدار میں یوریا (46% N) ڈالیں۔",
    "• **Excess Nitrogen:** Reduce nitrogenous fertilizer to avoid root burn and excessive leaf growth.": "• **زیادہ نائٹروجن:** جڑوں کو نقصان اور پتوں کی حد سے زیادہ بڑھوتری سے بچنے کے لیے نائٹروجنی کھاد کم کریں۔",
    "• **Phosphorus Deficiency:** Apply Single Super Phosphate (SSP) to support root development.": "• **فاسفورس کی کمی:** جڑوں کی نشوونما کے لیے سنگل سپر فاسفیٹ (SSP) ڈالیں۔",
    "• **Potassium Deficiency:** Apply Muriate of Potash (MOP) to improve disease resistance.": "• **پوٹاشیم کی کمی:** بیماریوں کے خلاف مزاحمت بہتر کرنے کے لیے میوریٹ آف پوٹاش (MOP) ڈالیں۔",
    "• **Acidic Soil:** Apply agricultural limestone to raise the pH.": "• **تیزابی مٹی:** pH بڑھانے کے لیے زرعی چونا استعمال کریں۔",
    "• **Alkaline Soil:** Apply elemental sulfur or gypsum to lower the pH.": "• **قلوی مٹی:** pH کم کرنے کے لیے عنصری سلفر یا جپسم استعمال کریں۔",
    "✅ Soil chemical composition is optimal for general crop cultivation.": "✅ عام فصلوں کے لیے مٹی کی کیمیائی ترکیب موزوں ہے۔",
    "Temperature": "درجہ حرارت",
    "Humidity": "نمی",
    "Rain (last 1h)": "بارش (گزشتہ 1 گھنٹہ)",
    "Smart Spray Weather Index": "اسپرے کے موسم کا اشاریہ",
    "✅ Favorable Spraying Window": "✅ اسپرے کے لیے موزوں وقت",
    "⚠️ High Rain/Heat Risk - Avoid Spraying": "⚠️ بارش یا گرمی کا زیادہ خطرہ — اسپرے سے گریز کریں",
    "⚠️ Marginal Conditions - Recheck Before Spraying": "⚠️ حالات مکمل موزوں نہیں — اسپرے سے پہلے دوبارہ جانچیں",
    "Rainfall, high temperature, or very high humidity increases spray wash-off or crop-stress risk.": "بارش، زیادہ درجہ حرارت یا بہت زیادہ نمی سے اسپرے بہنے یا فصل پر دباؤ کا خطرہ بڑھتا ہے۔",
    "Temperature, humidity, and recent rainfall are within the preferred range.": "درجہ حرارت، نمی اور حالیہ بارش اسپرے کے لیے موزوں حدود میں ہیں۔",
    "Some weather values are outside the preferred range; recheck conditions and the product label.": "موسم کی کچھ قدریں موزوں حدود سے باہر ہیں؛ حالات اور دوا کا لیبل دوبارہ دیکھیں۔",
    "Wind speed is not included in this index. Avoid spraying in windy conditions and follow the product label.": "اس اشاریے میں ہوا کی رفتار شامل نہیں۔ تیز ہوا میں اسپرے نہ کریں اور دوا کے لیبل پر عمل کریں۔",
    "🌧️ **Agronomic Insight:** Rain or heat risk detected. Delay chemical spraying to avoid wash-off or crop stress.": "🌧️ **زرعی مشورہ:** بارش یا گرمی کا خطرہ ہے۔ اسپرے بہنے یا فصل پر دباؤ سے بچنے کے لیے کیمیائی اسپرے مؤخر کریں۔",
    "🌦️ **Agronomic Insight:** Weather conditions are favorable for spraying. Low rainfall risk.": "🌦️ **زرعی مشورہ:** موسم اسپرے کے لیے موزوں ہے اور بارش کا امکان کم ہے۔",
    "**Agronomic Insight:** Conditions are mixed; check the product label and local wind before spraying.": "**زرعی مشورہ:** حالات ملے جلے ہیں؛ اسپرے سے پہلے دوا کا لیبل اور مقامی ہوا کی کیفیت دیکھیں۔",
    "Clear History": "تاریخ صاف کریں",
    "Primary ResNet50": "بنیادی ResNet50",
    "Hugging Face API": "Hugging Face API",
    "✅ **High Consensus:** Both models predict the same condition.": "✅ دونوں ماڈلز نے ایک ہی بیماری کی پیش گوئی کی ہے۔",
    "Both models identify the same crop, but their disease predictions differ. Verify symptoms manually.": "دونوں ماڈلز نے ایک ہی فصل شناخت کی ہے، مگر بیماری کے نتائج مختلف ہیں۔ علامات کی خود تصدیق کریں۔",
    "⚠️ **Secondary Disagreement:** The models predict different crops or conditions. Verify symptoms manually.": "⚠️ دونوں ماڈلز کے نتائج مختلف ہیں۔ علامات کی خود تصدیق کریں۔",
    "Hugging Face cross-check failed": "Hugging Face کی جانچ ناکام ہوئی",
    "Check that the model ID is correct and enabled for Inference Providers, and that HF_API_TOKEN has Inference Providers permission.": "ماڈل ID اور Inference Providers کی دستیابی چیک کریں، اور تصدیق کریں کہ HF_API_TOKEN کو Inference Providers کی اجازت حاصل ہے۔",
    "Enter crop-specific nutrient targets. Rates are kg/acre; soil-test values above are mg/kg and are not converted directly. ": "فصل کے مطابق غذائی ہدف درج کریں۔ شرح kg/acre میں ہے؛ اوپر کے مٹی ٹیسٹ mg/kg میں ہیں اور براہِ راست تبدیل نہیں ہوتے۔ ",
    "Marlas are converted at 160 marlas per acre. Calculations assume Urea 46-0-0, DAP 18-46-0, MOP 0-0-60, and 50 kg per bag.": "حساب میں 160 مرلے = 1 ایکڑ، یوریا 46-0-0، DAP 18-46-0، MOP 0-0-60 اور فی بوری 50 کلوگرام فرض کیے گئے ہیں۔",
    "Field area": "رقبہ",
    "Estimated Fertilizer Requirement": "کھاد کی اندازاً ضرورت",
    "to cover requirement": "ضرورت پوری کرنے کے لیے",
    "Regional Report": "علاقائی رپورٹ",
    "Fetching live weather for": "موسم کی تازہ معلومات حاصل ہو رہی ہیں:",
    "Could not fetch weather for": "موسم کی معلومات حاصل نہیں ہو سکیں:",
    "Condition": "حالت",
    "Calculated for": "حساب شدہ رقبہ:",
    "Urea is reduced by the nitrogen supplied through DAP. Confirm nutrient targets with local crop guidance.": "DAP سے ملنے والی نائٹروجن کو یوریا کی مقدار میں منہا کیا گیا ہے۔ غذائی اہداف کی مقامی زرعی ماہر سے تصدیق کریں۔",
    "There is no chemical cure — remove and destroy infected plants immediately. Disinfect tools between plants and control aphid/whitefly vectors to limit spread.": "اس وائرس کا کیمیائی علاج نہیں۔ متاثرہ پودے فوراً نکال کر تلف کریں، اوزار جراثیم سے پاک کریں اور بیماری پھیلانے والے کیڑوں کو قابو کریں۔",
    "🌧️ **Agronomic Insight:** Rain risk detected. Delay chemical spraying to avoid wash-off.": "🌧️ **زرعی مشورہ:** بارش کا امکان ہے۔ اسپرے بہہ جانے سے بچانے کے لیے کیمیائی اسپرے مؤخر کریں۔",
    "🌦️ **Agronomic Insight:** Weather conditions are favorable for spraying. Low rainfall risk.": "🌦️ **زرعی مشورہ:** اسپرے کے لیے موسم موزوں ہے اور بارش کا امکان کم ہے۔",
    "Weather-Based Crop & Sowing Advisory": "موسم کے مطابق فصل اور کاشت کا مشورہ",
    "Current Season": "موجودہ موسم",
    "Rabi Season": "ربیع کا موسم",
    "Kharif Season": "خریف کا موسم",
    "Optimal Major Crops to Sow Now": "ابھی کاشت کے لیے موزوں اہم فصلیں",
    "High-Value Winter Vegetables": "زیادہ منافع بخش سرمائی سبزیاں",
    "Smog & Fungal Risk Alert": "سموگ اور فنگس الرٹ",
    "Smog Alert": "سموگ الرٹ",
    "Fungal Risk Alert": "فنگس کا خطرہ",
    "Smog, haze, or dust is reported. Reduce prolonged field work and protect seedlings from dust.": "سموگ، دھند یا گردوغبار کی اطلاع ہے۔ کھیت میں طویل وقت کام محدود کریں اور ننھے پودوں کو گرد سے بچائیں۔",
    "No smog or dust condition is currently reported.": "اس وقت سموگ یا گردوغبار کی اطلاع نہیں ہے۔",
    "High humidity or rain may increase fungal pressure. Monitor leaves, avoid overhead irrigation, and maintain airflow.": "زیادہ نمی یا بارش سے فنگس کا خطرہ بڑھ سکتا ہے۔ پتوں پر نظر رکھیں، اوپر سے آبپاشی نہ کریں اور ہوا کی آمدورفت برقرار رکھیں۔",
    "Humidity and rainfall are not elevated; maintain routine leaf monitoring and avoid overwatering.": "نمی اور بارش زیادہ نہیں؛ پتوں کا معمول کے مطابق جائزہ لیتے رہیں اور ضرورت سے زیادہ پانی نہ دیں۔",
    "Temperature-based crop selection is a guide; verify planting dates, seed variety, and irrigation with local agricultural extension advice.": "درجہ حرارت کی بنیاد پر فصل کا انتخاب عمومی رہنمائی ہے؛ کاشت کے وقت، بیج کی قسم اور آبپاشی کی مقامی زرعی ماہر سے تصدیق کریں۔",
    "Network error while fetching weather data. Check your internet connection.": "موسم کی معلومات حاصل کرتے وقت نیٹ ورک کی خرابی ہوئی۔ انٹرنیٹ کنکشن چیک کریں۔",
    "There is no chemical cure — remove and destroy infected plants immediately. Disinfect tools between plants and control aphid/whitefly vectors to limit spread.": "اس وائرس کا کیمیائی علاج نہیں۔ متاثرہ پودے فوراً نکال کر تلف کریں، اوزار جراثیم سے پاک کریں اور بیماری پھیلانے والے کیڑوں کو قابو کریں۔",
}

URDU_WEATHER_CONDITIONS = {
    "clear sky": "صاف آسمان",
    "few clouds": "چند بادل",
    "scattered clouds": "بکھرے ہوئے بادل",
    "broken clouds": "جزوی ابر آلود",
    "overcast clouds": "مکمل ابر آلود",
    "light rain": "ہلکی بارش",
    "moderate rain": "درمیانی بارش",
    "heavy intensity rain": "تیز بارش",
    "very heavy rain": "بہت تیز بارش",
    "light snow": "ہلکی برف باری",
    "snow": "برف باری",
    "mist": "دھند",
    "fog": "گہری دھند",
    "haze": "فضا میں دھندلا پن",
    "thunderstorm": "گرج چمک کے ساتھ بارش",
}

URDU_DISEASE_NAMES = {
    "Apple___Apple_scab": "سیب - سیب کا اسکیب",
    "Apple___Black_rot": "سیب - سیاہ سڑن",
    "Apple___Cedar_apple_rust": "سیب - دیودار سیب کا زنگ",
    "Apple___healthy": "سیب - صحت مند",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "مکئی - سرمئی دھبوں کی بیماری",
    "Corn_(maize)___Common_rust_": "مکئی - عام زنگ",
    "Corn_(maize)___Northern_Leaf_Blight": "مکئی - شمالی پتوں کا جھلساؤ",
    "Corn_(maize)___healthy": "مکئی - صحت مند",
    "Grape___Black_rot": "انگور - سیاہ سڑن",
    "Grape___Esca_(Black_Measles)": "انگور - اسکا (سیاہ خسرہ)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "انگور - پتوں کا جھلساؤ",
    "Grape___healthy": "انگور - صحت مند",
    "Potato___Early_blight": "آلو - ابتدائی جھلساؤ",
    "Potato___Late_blight": "آلو - پچھیتی جھلساؤ",
    "Tomato___Bacterial_spot": "ٹماٹر - بیکٹیریائی دھبے",
    "Tomato___Late_blight": "ٹماٹر - پچھیتی جھلساؤ",
    "Tomato___Septoria_leaf_spot": "ٹماٹر - سیپٹوریا کے پتوں کے دھبے",
    "Tomato___Target_Spot": "ٹماٹر - ٹارگٹ دھبے",
    "Tomato___Tomato_mosaic_virus": "ٹماٹر - موزیک وائرس",
}

URDU_TREATMENT_DB = {
    "Apple___Apple_scab": "متاثرہ پتوں کو جمع کرکے تلف کریں، باغ میں ہوا کی آمدورفت بہتر رکھیں اور مقامی سفارش کے مطابق حفاظتی فنگس کش دوا استعمال کریں۔",
    "Apple___Black_rot": "متاثرہ شاخیں اور سوکھے پھل درخت اور زمین سے ہٹا کر تلف کریں۔ صفائی برقرار رکھیں اور مقامی ماہر کے مشورے سے حفاظتی فنگس کش دوا استعمال کریں۔",
    "Apple___Cedar_apple_rust": "اگر ممکن ہو تو قریبی جونیپر یا دیودار کے میزبان پودے ہٹائیں۔ کلیاں کھلنے کے وقت مقامی سفارش کے مطابق حفاظتی فنگس کش دوا استعمال کریں۔",
    "Apple___healthy": "بیماری کی علامت نہیں ملی۔ مناسب آبپاشی، شاخوں میں ہوا کی آمدورفت اور باغ کی صفائی برقرار رکھیں۔",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "فصلوں کی گردش کریں، مزاحم اقسام منتخب کریں اور متاثرہ باقیات کو تلف کریں۔ فنگس کش دوا کے لیے مقامی زرعی سفارش پر عمل کریں۔",
    "Corn_(maize)___Common_rust_": "مزاحم مکئی کی قسم لگائیں اور بیماری بڑھنے پر مقامی زرعی ماہر سے مناسب فنگس کش دوا کی تصدیق کریں۔",
    "Corn_(maize)___Northern_Leaf_Blight": "مزاحم قسم استعمال کریں اور متاثرہ فصل کی باقیات تلف کریں یا زمین میں دبا دیں۔ علامات بڑھنے پر مقامی ماہر سے مشورہ کریں۔",
    "Corn_(maize)___healthy": "فصل صحت مند دکھائی دیتی ہے۔ متوازن کھاد اور ضرورت کے مطابق آبپاشی جاری رکھیں۔",
    "Grape___Black_rot": "سوکھے متاثرہ انگور اور بیمار بیلیں کاٹ کر تلف کریں۔ بیلوں کی صفائی کریں اور مقامی سفارش کے مطابق حفاظتی فنگس کش دوا استعمال کریں۔",
    "Grape___Esca_(Black_Measles)": "خشک موسم میں متاثرہ لکڑی کاٹ کر تلف کریں اور اوزار صاف رکھیں۔ اس بیماری کا فوری علاج نہیں؛ بیل کو پانی اور غذائیت کا مناسب انتظام دیں۔",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "بیلوں میں ہوا کی آمدورفت بہتر بنانے کے لیے مناسب کانٹ چھانٹ کریں۔ متاثرہ پتے ہٹائیں اور مقامی سفارش کے مطابق علاج کریں۔",
    "Grape___healthy": "بیل صحت مند دکھائی دیتی ہے۔ مناسب کانٹ چھانٹ، آبپاشی اور متوازن غذائیت جاری رکھیں۔",
    "Potato___Early_blight": "متاثرہ پتے ہٹائیں، پودوں کے درمیان مناسب فاصلہ رکھیں اور فصلوں کی گردش کریں۔ فنگس کش دوا صرف لیبل اور مقامی سفارش کے مطابق استعمال کریں۔",
    "Potato___Late_blight": "یہ بیماری تیزی سے پھیل سکتی ہے۔ متاثرہ پودوں کو الگ کریں، شدید متاثرہ پتے تلف کریں اور فوری طور پر مقامی زرعی ماہر سے منظور شدہ علاج پوچھیں۔",
    "Tomato___Bacterial_spot": "متاثرہ پتے ہٹائیں، اوپر سے پانی دینے سے گریز کریں اور گیلے پودوں کو نہ چھوئیں۔ علاج کے لیے مقامی ماہر اور دوا کے لیبل سے رہنمائی لیں۔",
    "Tomato___Late_blight": "یہ بیماری تیزی سے پھیل سکتی ہے۔ متاثرہ پودوں کو الگ کریں، شدید متاثرہ حصے تلف کریں اور فوری طور پر مقامی زرعی ماہر سے منظور شدہ علاج پوچھیں۔",
    "Tomato___Septoria_leaf_spot": "نچلے متاثرہ پتے ہٹا کر تلف کریں، پودوں کے درمیان ہوا کی آمدورفت بہتر رکھیں اور مقامی سفارش کے مطابق علاج کریں۔",
    "Tomato___Target_Spot": "پودوں کے درمیان مناسب فاصلہ اور ہوا کی آمدورفت رکھیں۔ متاثرہ پتے ہٹائیں اور مناسب علاج کے لیے مقامی زرعی مشورہ لیں۔",
    "Tomato___Tomato_mosaic_virus": "اس وائرس کا کیمیائی علاج نہیں۔ متاثرہ پودے فوراً نکال کر تلف کریں، اوزار جراثیم سے پاک کریں اور بیماری پھیلانے والے کیڑوں کو قابو کریں۔",
}

URDU_ASSISTANT_RESPONSES = {
    "black_rot_clarify": "سیاہ سڑن سیب اور انگور دونوں میں ہو سکتی ہے۔ آپ کس فصل کے بارے میں پوچھ رہے ہیں؟",
    "label_safety": "کسی بھی دوا کو صرف اس کے لیبل اور مقامی زرعی ماہر کی ہدایات کے مطابق استعمال کریں۔",
    "spray": "اسپرے کا انتخاب فصل اور بیماری کی درست شناخت پر منحصر ہے۔ دوا کا لیبل پڑھیں، حفاظتی سامان پہنیں، تیز ہوا میں اسپرے نہ کریں، اور بارش متوقع ہو تو اسپرے مؤخر کریں۔ فصل اور علامات بتائیں تو میں زیادہ موزوں مشورہ دوں گا۔",
    "soil": "مٹی کے ٹیسٹ کے لیے Soil Advisory اور کھاد کی مقدار کے لیے NPK Fertilizer Requirement Calculator استعمال کریں۔ فصل کے مطابق N، P₂O₅ اور K₂O کے ہدف kg/acre میں درج کریں۔ صرف mg/kg ٹیسٹ سے کھاد کی درست مقدار طے نہیں ہوتی؛ فصل اور مٹی کی حالت بھی اہم ہیں۔",
    "weather": "بارش اسپرے کو دھو سکتی ہے اور زیادہ نمی بعض بیماریوں کو بڑھا سکتی ہے۔ اسپرے سے پہلے Weather Engine دیکھیں، دوا کے لیبل پر درج بارش سے پہلے کا وقفہ اپنائیں، اور تیز ہوا میں اسپرے نہ کریں۔",
    "disease": "تشخیص کے لیے متاثرہ پتے کی صاف، قریب سے لی گئی تصویر اپ لوڈ کریں۔ ماڈل کا نتیجہ ابتدائی رہنمائی ہے، حتمی تشخیص نہیں؛ فصل، علامات کا پھیلاؤ اور موسم بھی دیکھیں۔",
    "fallback": "میں فصل کی بیماریوں، اسپرے کی حفاظت، مٹی کی غذائیت اور موسم کے بارے میں مدد کر سکتا ہوں۔ فصل کا نام، علامات، بڑھوتری کا مرحلہ اور اپنا سوال بتائیں۔",
}

ui_language = st.sidebar.radio(
    "Language / زبان",
    ["English", "اردو"],
    horizontal=True,
    key="ui_language"
)


def tr(text: str) -> str:
    return URDU_UI.get(text, text) if ui_language == "اردو" else text


def display_disease_name(raw_class: str, english_name: str) -> str:
    if ui_language == "اردو":
        return URDU_DISEASE_NAMES.get(raw_class, english_name)
    return english_name

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-12px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .stApp { background: linear-gradient(180deg, #F6FBF6 0%, #FFFFFF 250px); }

    .hero {
        background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(46,125,50,0.25);
        animation: fadeIn 0.7s ease-out;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        opacity: 0.92;
        margin-top: 0.3rem;
        color: #EAF7EA;
    }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-top: 10px;
        transition: background 0.2s ease-in-out, transform 0.2s ease-in-out;
    }
    .badge:hover {
        background: rgba(255,255,255,0.32);
        transform: translateY(-1px);
    }

    div[data-testid="stMetricValue"] { font-size: 1.3rem; color: #2E7D32; }

    .result-card {
        background: white;
        border: 1px solid #E3EEE3;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 0.9rem;
        transition: box-shadow 0.25s ease-in-out, transform 0.25s ease-in-out;
        animation: fadeIn 0.5s ease-out;
    }
    .result-card:hover {
        box-shadow: 0 8px 20px rgba(46,125,50,0.14);
        transform: translateY(-3px);
    }

    /* Tabs: smoother switch + hover color */
    div[data-testid="stTabs"] button {
        transition: color 0.2s ease-in-out;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button:hover {
        color: #2E7D32 !important;
    }
    div[data-testid="stTabs"] > div > div[data-baseweb="tab-panel"] {
        animation: fadeIn 0.4s ease-out;
    }

    /* File uploader glow on hover */
    div[data-testid="stFileUploaderDropzone"] {
        transition: border-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2E7D32 !important;
        box-shadow: 0 0 0 3px rgba(46,125,50,0.1);
    }

    /* Buttons: lift + slight scale on hover */
    .stButton button {
        transition: all 0.2s ease-in-out;
        border-radius: 8px;
    }
    .stButton button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 4px 12px rgba(46,125,50,0.18);
    }

    /* Sidebar items slide in */
    section[data-testid="stSidebar"] .element-container {
        animation: slideIn 0.4s ease-out;
    }

    /* Expander (Hugging Face second opinion) subtle entrance */
    div[data-testid="stExpander"] {
        animation: fadeIn 0.5s ease-out;
        border-radius: 10px;
    }

    .footer-note {
        text-align: center;
        color: #9AA5A0;
        font-size: 0.78rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <p class="hero-title">{tr("🌱 AgriSmart AI Decision Support System")}</p>
    <p class="hero-subtitle">{tr("Plant Pathology Diagnosis · Soil Chemistry Advisory · Regional Weather Insights")}</p>
    <span class="badge">{tr("🎯 97.36% Validation Accuracy")}</span>
    <span class="badge">{tr("🧬 ResNet50 Fine-Tuned")}</span>
    <span class="badge">{tr("🌾 19 Disease Classes")}</span>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL (cached so it loads only once)
# =========================================================
# ==============================================================================
# LOAD MODEL (cached so it loads only once)
# ==============================================================================
@st.cache_resource
def load_disease_model():
    try:
        with open('resnet_plant_disease_model.pkl', 'rb') as f:
            data = pickle.load(f)
            
        # Agar dictionary format mein save hua hai
        if isinstance(data, dict):
            return data.get('model'), data.get('class_indices', {})
        # Agar direct keras model object saved hai
        else:
            return data, {}
    except Exception as e:
        return None, None

disease_model, class_indices = load_disease_model()

# =========================================================
# TREATMENT DATABASE — matches EXACTLY the 19 trained classes
# =========================================================
treatment_db = {
    "Apple___Apple_scab": "Apply protective fungicides such as Captan or Mancozeb during early bloom. Rake and destroy fallen leaves to eliminate overwintering fungal spores.",
    "Apple___Black_rot": "Prune out infected twigs and remove mummified fruit from the tree and ground. Apply copper-based fungicides on a regular schedule through the growing season.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper/cedar trees where feasible, since they host the fungus. Apply protective fungicides (myclobutanil or copper-based) starting at bud break.",
    "Apple___healthy": "No disease detected. Maintain a consistent irrigation schedule and good canopy ventilation to keep the tree in this healthy state.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate with non-host crops and use resistant hybrids where available. Apply strobilurin or triazole fungicides if lesions appear before tasseling.",
    "Corn_(maize)___Common_rust_": "Plant rust-resistant hybrids. Apply foliar fungicides containing azoxystrobin or propiconazole at the first sign of orange pustules.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids and practice deep tillage to bury infected residue. Apply fungicides early if the disease appears before flowering.",
    "Corn_(maize)___healthy": "Crop is in healthy condition. Maintain balanced nitrogen application, especially during the silking stage.",

    "Grape___Black_rot": "Remove mummified berries and infected canes during dormant pruning. Apply protectant fungicides (mancozeb or myclobutanil) starting at early shoot growth.",
    "Grape___Esca_(Black_Measles)": "Prune out and destroy infected wood during dry weather. There is no curative spray — focus on vine stress reduction and sanitation.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve canopy airflow through proper leaf pulling. Apply copper-based fungicides at the first sign of leaf spotting.",
    "Grape___healthy": "Vine is healthy. Continue routine canopy management and balanced fertilization.",

    "Potato___Early_blight": "Apply protective fungicides such as chlorothalonil or mancozeb. Maintain adequate plant spacing and practice a 3-year crop rotation.",
    "Potato___Late_blight": "Apply systemic fungicides containing metalaxyl or cymoxanil immediately. Destroy severely infected foliage to prevent spore spread — this disease can devastate a field quickly.",

    "Tomato___Bacterial_spot": "Apply copper-based bactericides mixed with mancozeb. Avoid overhead irrigation and working in fields when plants are wet.",
    "Tomato___Late_blight": "Apply preventive copper-based fungicide sprays immediately and improve field air drainage. Remove and destroy severely affected vines to halt spread.",
    "Tomato___Septoria_leaf_spot": "Remove and destroy lower infected leaves. Apply chlorothalonil-based fungicides on a 7-10 day schedule during humid conditions.",
    "Tomato___Target_Spot": "Improve plant spacing and airflow. Apply protectant fungicides (chlorothalonil or azoxystrobin) at early symptom onset.",
    "Tomato___Tomato_mosaic_virus": "There is no chemical cure — remove and destroy infected plants immediately. Disinfect tools between plants and control aphid/whitefly vectors to limit spread."
}

DEFAULT_HF_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


def query_huggingface_second_opinion(image_bytes, hf_token, model_id):
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/octet-stream"
    }
    try:
        response = requests.post(api_url, headers=headers, data=image_bytes, timeout=30)
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            payload = {"error": response.text[:500] or response.reason}

        if response.ok:
            return payload, None

        if isinstance(payload, dict):
            error = payload.get("error", str(payload))
        else:
            error = str(payload)
        return None, f"HTTP {response.status_code}: {error}"
    except requests.exceptions.RequestException as exc:
        return None, f"Network error: {exc}"

def get_severity(disease_name: str) -> str:
    """Simple heuristic to badge the severity of a detected condition."""
    if "healthy" in disease_name.lower():
        return "✅ Healthy"
    if "mosaic_virus" in disease_name.lower() or "late_blight" in disease_name.lower() or "esca" in disease_name.lower():
        return "🔴 High Risk"
    return "🟠 Moderate Risk"


def get_agri_assistant_response(question: str) -> str:
    query = re.sub(r"[^\w]+", " ", question.lower(), flags=re.UNICODE).strip()
    disease_aliases = {
        "apple scab": ["Apple___Apple_scab"],
        "سیب کا اسکیب": ["Apple___Apple_scab"],
        "black rot": ["Apple___Black_rot", "Grape___Black_rot"],
        "سیاہ سڑن": ["Apple___Black_rot", "Grape___Black_rot"],
        "cedar apple rust": ["Apple___Cedar_apple_rust"],
        "دیودار سیب کا زنگ": ["Apple___Cedar_apple_rust"],
        "gray leaf spot": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"],
        "grey leaf spot": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"],
        "سرمئی دھبوں کی بیماری": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"],
        "common rust": ["Corn_(maize)___Common_rust_"],
        "عام زنگ": ["Corn_(maize)___Common_rust_"],
        "northern leaf blight": ["Corn_(maize)___Northern_Leaf_Blight"],
        "شمالی پتوں کا جھلساؤ": ["Corn_(maize)___Northern_Leaf_Blight"],
        "esca": ["Grape___Esca_(Black_Measles)"],
        "black measles": ["Grape___Esca_(Black_Measles)"],
        "اسکا": ["Grape___Esca_(Black_Measles)"],
        "grape leaf blight": ["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"],
        "انگور کے پتوں کا جھلساؤ": ["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"],
        "early blight": ["Potato___Early_blight"],
        "ابتدائی جھلساؤ": ["Potato___Early_blight"],
        "late blight": ["Potato___Late_blight", "Tomato___Late_blight"],
        "پچھیتی جھلساؤ": ["Potato___Late_blight", "Tomato___Late_blight"],
        "bacterial spot": ["Tomato___Bacterial_spot"],
        "بیکٹیریائی دھبے": ["Tomato___Bacterial_spot"],
        "septoria": ["Tomato___Septoria_leaf_spot"],
        "سیپٹوریا": ["Tomato___Septoria_leaf_spot"],
        "target spot": ["Tomato___Target_Spot"],
        "ٹارگٹ دھبے": ["Tomato___Target_Spot"],
        "mosaic virus": ["Tomato___Tomato_mosaic_virus"],
        "موزیک وائرس": ["Tomato___Tomato_mosaic_virus"],
    }

    matched_classes = {
        class_name
        for phrase, class_names in disease_aliases.items()
        if phrase in query
        for class_name in class_names
    }
    crop_aliases = {
        "apple": ("apple", "سیب"),
        "corn": ("corn", "maize", "مکئی"),
        "grape": ("grape", "انگور"),
        "potato": ("potato", "آلو"),
        "tomato": ("tomato", "ٹماٹر"),
    }
    crops_in_question = {
        crop for crop, aliases in crop_aliases.items()
        if any(alias in query for alias in aliases)
    }
    if crops_in_question:
        matched_classes = {
            class_name for class_name in matched_classes
            if any(crop in class_name.lower() for crop in crops_in_question)
        }

    if matched_classes:
        if len(matched_classes) > 1:
            if ui_language == "اردو":
                return URDU_ASSISTANT_RESPONSES["black_rot_clarify"]
            return "Black rot affects both apple and grape in this app. Which crop are you asking about?"
        class_name = next(iter(matched_classes))
        crop_name, disease_name = class_name.split("___", maxsplit=1)
        treatment = treatment_db.get(class_name)
        if ui_language == "اردو":
            urdu_treatment = URDU_TREATMENT_DB.get(class_name, URDU_ASSISTANT_RESPONSES["disease"])
            disease_label = URDU_DISEASE_NAMES.get(class_name, disease_name.replace("_", " "))
            return f"**{disease_label}**: {urdu_treatment} {URDU_ASSISTANT_RESPONSES['label_safety']}"
        return (
            f"For **{crop_name.replace('_', ' ')} {disease_name.replace('_', ' ')}**: {treatment} "
            "Follow the product label and local agricultural guidance for approved products and application rates."
        )

    if any(term in query for term in ("spray", "fungicide", "pesticide", "chemical", "when to spray", "kab spray", "chhirkao", "اسپرے", "چھڑکاؤ", "فنگس کش", "کیڑے مار")):
        if ui_language == "اردو":
            return URDU_ASSISTANT_RESPONSES["spray"]
        return (
            "Spray choice and timing depend on the crop and confirmed disease. Check the product label, wear the listed "
            "protective equipment, and avoid spraying in wind or when rain is expected soon. Tell me the crop and symptoms "
            "for more targeted guidance."
        )

    if any(term in query for term in ("soil", "nutrient", "fertilizer", "fertiliser", "urea", "dap", "potash", "ph", "npk", "mitti", "khad", "khaad", "zameen", "مٹی", "کھاد", "یوریا", "فاسفورس", "پوٹاش", "نائٹروجن", "غذائیت")):
        if ui_language == "اردو":
            return URDU_ASSISTANT_RESPONSES["soil"]
        return (
            "Use the Soil Advisory for lab values and the NPK Fertilizer Requirement Calculator for crop-specific target "
            "rates. Enter N, P₂O₅, and K₂O targets in kg/acre; the calculator accounts for nitrogen supplied by DAP. "
            "Soil-test mg/kg values alone do not determine a fertilizer dose without crop and soil context."
        )

    if any(term in query for term in ("rain", "weather", "humidity", "spraying conditions", "barish", "mausam", "baarish", "موسم", "بارش", "نمی")):
        if ui_language == "اردو":
            return URDU_ASSISTANT_RESPONSES["weather"]
        return (
            "Rain can wash off a spray, and high humidity can favor some leaf diseases. Check the Weather Engine before "
            "spraying, follow the product label's rain-free interval, and avoid treating crops in strong wind."
        )

    if any(term in query for term in ("disease", "symptom", "leaf", "spot", "yellow", "diagnos", "bimari", "beemari", "alamat", "patta", "patte", "بیماری", "علامات", "پتے", "دھبے", "پیلا")):
        if ui_language == "اردو":
            return URDU_ASSISTANT_RESPONSES["disease"]
        return (
            "Upload a clear leaf image in the diagnosis panel for a model prediction. It is a screening aid, not a "
            "confirmed diagnosis; crop, symptom spread, and growing conditions help distinguish similar diseases."
        )

    if ui_language == "اردو":
        return URDU_ASSISTANT_RESPONSES["fallback"]
    return (
        "I can help with common crop diseases, spray safety, soil nutrients, and weather. Share the crop, visible "
        "symptoms, growth stage, and your question so I can point you to the most relevant guidance."
    )


def get_weather_sowing_advice(
    temp: float,
    humidity: int,
    month: int | None = None
) -> tuple[str, str, list[tuple[str, str]], list[tuple[str, str]], str]:
    current_month = month if month is not None else datetime.now().month
    is_rabi = current_month in (10, 11, 12, 1, 2, 3)
    is_urdu = ui_language == "اردو"

    if is_rabi:
        season = tr("Rabi Season")
        season_period = "اکتوبر تا مارچ" if is_urdu else "Oct–Mar"
        if 12 <= temp <= 25:
            major_crops = [
                ("🌾 گندم" if is_urdu else "🌾 Wheat", "موزوں مقامی قسم منتخب کرکے مناسب نمی والی تیار زمین میں بروقت بوائی کریں۔" if is_urdu else "Sow a locally recommended variety in a prepared, adequately moist seedbed."),
                ("🌻 سرسوں" if is_urdu else "🌻 Mustard", "مقامی سفارش کے مطابق قسم اور بوائی کا وقت منتخب کریں۔" if is_urdu else "Choose a locally recommended variety and sowing date."),
            ]
        elif temp > 25:
            major_crops = [
                ("🌾 گندم" if is_urdu else "🌾 Wheat", "ربیع کی اہم فصل ہے؛ موجودہ درجہ حرارت قدرے گرم ہے، اس لیے مقامی بوائی کے وقت کی تصدیق کریں۔" if is_urdu else "A key Rabi crop; current temperatures are warm, so confirm the local sowing window."),
                ("🌻 سرسوں" if is_urdu else "🌻 Mustard", "ربیع کی اہم فصل ہے؛ کھیت تیار کریں اور مقامی زرعی مشورے کے مطابق بوائی کریں۔" if is_urdu else "A key Rabi crop; prepare the field and follow local sowing advice."),
            ]
        else:
            major_crops = [
                ("🌾 گندم" if is_urdu else "🌾 Wheat", "ربیع میں کاشت ہو سکتی ہے؛ دیر سے بوائی کے لیے موزوں قسم مقامی مشورے سے منتخب کریں۔" if is_urdu else "Still fits the Rabi calendar; confirm a suitable late-sowing variety locally."),
                ("🌻 سرسوں" if is_urdu else "🌻 Mustard", "بوائی کی موزوں قسم اور وقت کے لیے مقامی زرعی ماہر سے مشورہ کریں۔" if is_urdu else "Check local guidance for a suitable variety and sowing date."),
            ]

        if temp <= 26:
            winter_vegetables = [
                ("🥔 آلو" if is_urdu else "🥔 Potato", "صحت مند بیج اور نکاسی والی زمین استعمال کریں۔" if is_urdu else "Use healthy seed and well-drained soil."),
                ("🫛 مٹر" if is_urdu else "🫛 Peas", "ٹھنڈے موسم میں بوائی کریں اور مٹی میں مناسب نمی رکھیں۔" if is_urdu else "Sow in cool weather and maintain even soil moisture."),
                ("🧄 لہسن" if is_urdu else "🧄 Garlic", "صحت مند جوئے نکاسی والی زمین میں لگائیں۔" if is_urdu else "Plant healthy cloves in well-drained soil."),
            ]
        else:
            winter_vegetables = [
                ("🥔 آلو" if is_urdu else "🥔 Potato", "موزوں ٹھنڈے وقت تک کیاریاں تیار کریں اور مقامی بوائی کی تاریخ دیکھیں۔" if is_urdu else "Prepare beds and confirm local timing before planting in cooler weather."),
                ("🫛 مٹر" if is_urdu else "🫛 Peas", "موجودہ گرمی میں انتظار کریں؛ ٹھنڈے موسم میں بوائی زیادہ موزوں ہے۔" if is_urdu else "Wait for cooler weather; current temperatures are warm for sowing."),
                ("🧄 لہسن" if is_urdu else "🧄 Garlic", "مقامی سفارش کے مطابق ٹھنڈے موسم میں جوئے لگائیں۔" if is_urdu else "Plant cloves in cooler weather following local guidance."),
            ]
    else:
        season = tr("Kharif Season")
        season_period = "اپریل تا ستمبر" if is_urdu else "Apr–Sep"
        if 20 <= temp <= 35:
            major_crops = [
                ("🌽 مکئی" if is_urdu else "🌽 Maize", "مقامی خریف کیلنڈر کے مطابق کاشت کریں۔" if is_urdu else "Sow according to the local Kharif calendar."),
                ("☁️ کپاس" if is_urdu else "☁️ Cotton", "مقامی سفارش کردہ قسم اور دستیاب آبپاشی کو مدنظر رکھیں۔" if is_urdu else "Use a locally recommended variety and ensure irrigation is available."),
                ("🌾 دھان" if is_urdu else "🌾 Rice", "صرف وہاں کاشت کریں جہاں پانی، زمین اور مقامی وقت موزوں ہو۔" if is_urdu else "Plant only where water supply, soil, and local timing are suitable."),
            ]
        elif temp > 35:
            major_crops = [
                ("🌽 مکئی" if is_urdu else "🌽 Maize", "شدید گرمی میں پودوں کو دباؤ ہو سکتا ہے؛ آبپاشی یقینی بنائیں۔" if is_urdu else "Heat stress is possible; ensure reliable irrigation."),
                ("☁️ کپاس" if is_urdu else "☁️ Cotton", "بوائی سے پہلے مقامی موسم اور پانی کی دستیابی دیکھیں۔" if is_urdu else "Check local weather and water availability before sowing."),
            ]
        else:
            major_crops = [
                ("🌽 مکئی" if is_urdu else "🌽 Maize", "موجودہ درجہ حرارت میں پودوں کی ابتدائی بڑھوتری سست ہو سکتی ہے؛ مقامی وقت کی تصدیق کریں۔" if is_urdu else "Cool conditions may slow establishment; confirm the local sowing window."),
                ("☁️ کپاس" if is_urdu else "☁️ Cotton", "موزوں درجہ حرارت آنے تک بوائی مؤخر کرنے پر غور کریں۔" if is_urdu else "Consider waiting for a more suitable temperature before sowing."),
            ]
        winter_vegetables = [
            ("🗓️ سرمائی سبزیاں" if is_urdu else "🗓️ Winter vegetables", "ابھی ان کی بنیادی بوائی کا وقت نہیں؛ ربیع کے موسم کے لیے بیج اور کیاریاں تیار کریں۔" if is_urdu else "Outside their main sowing window; prepare seed and beds for Rabi."),
        ]

    if humidity >= 80:
        moisture_note = "زیادہ نمی میں پھپھوندی کا خطرہ بڑھ سکتا ہے؛ پانی کھڑا نہ ہونے دیں، ہوا کی آمدورفت رکھیں اور پتوں کے دھبوں پر نظر رکھیں۔" if is_urdu else "High humidity can increase fungal risk; avoid waterlogged beds, allow airflow, and monitor for leaf spots."
    elif humidity < 40:
        moisture_note = "نمی کم ہے؛ اگاؤ کے دوران زمین میں یکساں نمی رکھیں اور ننھے پودوں کو خشک نہ ہونے دیں۔" if is_urdu else "Humidity is low; keep the seedbed evenly moist during germination and protect seedlings from drying out."
    else:
        moisture_note = "نمی معتدل ہے؛ مٹی میں یکساں نمی رکھیں اور ضرورت سے زیادہ پانی نہ دیں۔" if is_urdu else "Humidity is moderate; maintain even soil moisture and avoid overwatering."

    return season, season_period, major_crops, winter_vegetables, moisture_note


def get_spray_weather_status(temp: float, humidity: int, rain_mm: float) -> str:
    if rain_mm > 0 or humidity >= 85 or temp >= 35:
        return "high_risk"
    if rain_mm == 0 and 40 <= humidity <= 75 and 10 <= temp <= 30:
        return "favorable"
    return "marginal"


def build_diagnosis_pdf(
    condition: str,
    confidence: float,
    severity: str,
    treatment: str,
    generated_at: datetime
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="AgriSmart AI Diagnosis Report",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="AgriReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#2E7D32"),
        fontSize=20,
        leading=24,
        alignment=0,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="AgriReportSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#2E7D32"),
        fontSize=13,
        leading=17,
        spaceBefore=14,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="AgriReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
    ))

    severity_label = {
        "✅ Healthy": "Healthy",
        "🔴 High Risk": "High Risk",
        "🟠 Moderate Risk": "Moderate Risk",
    }.get(severity, severity)
    details = [
        [Paragraph("<b>Detected condition</b>", styles["AgriReportBody"]), Paragraph(condition, styles["AgriReportBody"])],
        [Paragraph("<b>Confidence</b>", styles["AgriReportBody"]), Paragraph(f"{confidence:.2f}%", styles["AgriReportBody"])],
        [Paragraph("<b>Severity</b>", styles["AgriReportBody"]), Paragraph(severity_label, styles["AgriReportBody"])],
        [Paragraph("<b>Generated</b>", styles["AgriReportBody"]), Paragraph(generated_at.strftime("%Y-%m-%d %H:%M:%S"), styles["AgriReportBody"])],
    ]
    details_table = Table(details, colWidths=[1.55 * inch, 5.65 * inch], hAlign="LEFT")
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF6EF")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#26352A")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6E5D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    story = [
        Paragraph("AgriSmart AI", styles["AgriReportTitle"]),
        Paragraph("Plant Disease Diagnosis Report", styles["Heading2"]),
        Spacer(1, 12),
        details_table,
        Paragraph("Recommended Action Plan", styles["AgriReportSection"]),
        Paragraph(treatment, styles["AgriReportBody"]),
        Spacer(1, 12),
        Paragraph("For agricultural guidance, follow local extension recommendations and product labels.", styles["Italic"]),
    ]
    document.build(story)
    return buffer.getvalue()


def create_disease_severity_overlay(image: Image.Image) -> tuple[Image.Image, bool]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    green_leaf = (green > red * 1.04) & (green > blue * 1.02) & (green > 45)
    leaf_context = np.asarray(
        Image.fromarray(green_leaf.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(31))
    ) > 0

    brown_spots = (red > green * 1.12) & (red > blue * 1.15)
    yellow_spots = (red > green * 1.02) & (green > blue * 1.12) & (red - blue > 25)
    spot_mask = (brown_spots | yellow_spots) & leaf_context
    if not np.any(spot_mask):
        return image.convert("RGB"), False

    mask = Image.fromarray(spot_mask.astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2))

    overlay = Image.new("RGBA", image.size, (230, 35, 15, 0))
    overlay.putalpha(mask.point(lambda value: int(value * 0.55)))
    result = Image.alpha_composite(image.convert("RGBA"), overlay)

    contour = mask.point(lambda value: 255 if value > 80 else 0).filter(ImageFilter.FIND_EDGES)
    contour = contour.point(lambda value: 220 if value > 24 else 0)
    outline = Image.new("RGBA", image.size, (255, 220, 0, 0))
    outline.putalpha(contour)
    result = Image.alpha_composite(result, outline)
    return result.convert("RGB"), True



# =========================================================
# SESSION STATE — keeps a history of diagnoses in this session
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header(tr("📊 Session Summary"))
    st.metric(tr("Diagnoses Run"), len(st.session_state.history))
    if disease_model is not None:
        num_classes = len(class_indices) if class_indices else "38"
        st.success(f"{tr('Model loaded')} 🎯 ({num_classes} {tr('classes')})")
    else:
        st.error(tr("Model failed to load"))

    st.markdown("---")
    st.subheader(tr("🕘 Recent History"))
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-5:]):
            history_name = display_disease_name(entry.get("raw_class", ""), entry["disease"])
            st.caption(f"{entry['time']} — **{history_name}** ({entry['confidence']:.1f}%)")
    else:
        st.caption(tr("No diagnoses yet this session."))

    st.markdown("---")
    if st.button(tr("🗑️ Clear History")):
        st.session_state.history = []
        st.rerun()

# =========================================================
# NAVIGATION TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    tr("🍃 Leaf Disease Diagnosis"),
    tr("🧪 Soil Advisory"),
    tr("🌤️ Weather Engine")
])

# ---------------------------------------------------------
# TAB 1: LEAF DISEASE DIAGNOSIS
# ---------------------------------------------------------
with tab1:
    st.subheader(tr("Plant Leaf Pathogen Identification"))
    st.caption(tr("Upload a leaf image to run AI-powered disease classification and get an actionable treatment plan."))

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_file = st.file_uploader(tr("Upload Leaf Image"), type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption=tr("Uploaded Leaf Image"), use_container_width=True)
            if st.checkbox(tr("🔍 Show Disease Severity Overlay"), key="show_disease_severity_overlay"):
                overlay_image, spots_highlighted = create_disease_severity_overlay(img)
                st.image(
                    overlay_image,
                    caption=tr("Estimated Disease Severity Overlay"),
                    use_container_width=True
                )
                st.caption(tr(
                    "Red overlay marks likely discolored spots; it is a visual estimate, not confirmed disease segmentation."
                ))
                if not spots_highlighted:
                    st.info(tr("No obvious discolored spots were highlighted; this does not rule out disease."))

    with col2:
        if uploaded_file is not None:
            if disease_model is None:
                st.error(tr("Model could not be loaded. Check that resnet_plant_disease_model.pkl is in the app folder."))
            else:
                with st.spinner(tr("Analyzing leaf image...")):
                    # --- Preprocessing MUST match training exactly ---
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized).astype("float32")
                    img_array = preprocess_input(img_array)          # same as training (NOT /255.0)
                    img_array = np.expand_dims(img_array, axis=0)

                    preds = disease_model.predict(img_array, verbose=0)
                    pred_idx = np.argmax(preds[0])
                    confidence = float(np.max(preds[0])) * 100

                    labels_map = {v: k for k, v in class_indices.items()}
                    raw_class = labels_map.get(pred_idx, f"Class_{pred_idx}")
                    clean_name = raw_class.replace("___", " - ").replace("_", " ")
                    display_name = display_disease_name(raw_class, clean_name)
                    severity = get_severity(raw_class)

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"#### {tr('📊 Diagnostic Results')}")
                m1, m2 = st.columns(2)
                m1.metric(tr("Identified Condition"), display_name)
                m2.metric(tr("Confidence"), f"{confidence:.2f}%")
                st.progress(min(int(confidence), 100), text=f"{tr('Model Confidence')}: {confidence:.1f}%")
                if confidence < 60.0:
                    st.warning(tr(
                        "Low Confidence Detection: The uploaded leaf image may belong to an unsupported crop or symptoms are unclear. Please upload a clear, close-up photo of an affected leaf."
                    ))
                st.markdown(f"**{tr('Severity')}:** {tr(severity)}")
                st.markdown('</div>', unsafe_allow_html=True)

                severity_breakdown = {
                    "✅ Healthy": (0, "Low", 10, "Routine", 10),
                    "🟠 Moderate Risk": (35, "Moderate", 50, "Prompt", 60),
                    "🔴 High Risk": (70, "High", 90, "Immediate", 100),
                }
                affected_area, spread_label, spread_score, urgency_label, urgency_score = severity_breakdown[severity]

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"#### {tr('Visual Breakdown')}")
                st.caption(tr("Severity-based estimate"))
                breakdown_columns = st.columns(3)
                breakdown_columns[0].metric(
                    tr("Estimated Affected Area (%)"),
                    f"~{affected_area}%"
                )
                breakdown_columns[0].progress(affected_area, text=f"{affected_area}%")
                breakdown_columns[1].metric(
                    tr("Spread Risk Level"),
                    tr(spread_label)
                )
                breakdown_columns[1].progress(spread_score, text=f"{spread_score}%")
                breakdown_columns[2].metric(
                    tr("Urgency of Treatment"),
                    tr(urgency_label)
                )
                breakdown_columns[2].progress(urgency_score, text=f"{urgency_score}%")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"#### {tr('📋 Recommended Action Plan')}")
                treatment = treatment_db.get(
                    raw_class,
                    "Monitor the crop closely and consult a local agricultural extension officer for targeted treatment guidance."
                )
                st.info(treatment)
                audio_key = f"{raw_class}|{confidence:.4f}|{treatment}"
                if st.button(tr("🔊 Generate audio report"), key="generate_diagnosis_audio"):
                    audio_text = (
                        f"Diagnosis: {clean_name}. Confidence: {confidence:.1f} percent. "
                        f"Severity: {get_severity(raw_class)}. Recommended action plan: {treatment}"
                    )
                    try:
                        audio_buffer = BytesIO()
                        with st.spinner(tr("Generating audio report...")):
                            gTTS(text=audio_text, lang="en").write_to_fp(audio_buffer)
                        st.session_state.diagnosis_audio = {
                            "key": audio_key,
                            "data": audio_buffer.getvalue(),
                        }
                    except Exception as exc:
                        st.error(f"{tr('Could not generate audio. Check your internet connection and try again.')} ({exc})")

                saved_audio = st.session_state.get("diagnosis_audio", {})
                if saved_audio.get("key") == audio_key and saved_audio.get("data"):
                    st.audio(saved_audio["data"], format="audio/mpeg")
                st.markdown('</div>', unsafe_allow_html=True)

                # --- reaction animation (moved here, after raw_class exists) ---
                if "healthy" in raw_class.lower():
                    st.balloons()
                elif get_severity(raw_class) == "🔴 High Risk":
                    st.toast(tr("⚠️ High-risk disease detected — review treatment plan below"), icon="🔴")
                else:
                    st.toast(tr("✅ Diagnosis complete"), icon="🌱")

                          
                # ---- Optional: Hugging Face second opinion (cross-check) ----
                st.markdown("---")
                st.markdown(tr("#### 🤖 Ensemble Intelligence (Automated HF API Cross-Check)"))

                try:
                    hf_token = st.secrets.get("HF_API_TOKEN", "")
                    hf_model_id = st.secrets.get(
                        "HF_MODEL_ID", DEFAULT_HF_MODEL_ID
                    )
                except Exception:
                    hf_token = ""
                    hf_model_id = DEFAULT_HF_MODEL_ID

                if hf_token:
                    uploaded_file.seek(0)
                    img_bytes = uploaded_file.read()

                    with st.spinner("Calling Hugging Face API for automated cross-check..."):
                        hf_result, hf_error = query_huggingface_second_opinion(
                            img_bytes, hf_token, hf_model_id
                        )

                    if isinstance(hf_result, list) and hf_result and "label" in hf_result[0]:
                        top = hf_result[0]
                        hf_label = top["label"].replace("___", " - ").replace("_", " ")
                        hf_score = top["score"] * 100

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.info(f"**{tr('Primary ResNet50')}:**\n\n{display_name} ({confidence:.1f}%)")
                        with col_b:
                            st.success(f"**{tr('Hugging Face API')}:**\n\n{hf_label} ({hf_score:.1f}%)")

                        primary_crop = clean_name.split("-")[0].strip().lower()
                        hf_crop = hf_label.split("-")[0].strip().lower()
                        if clean_name.lower() == hf_label.lower():
                            st.success(tr("✅ **High Consensus:** Both models predict the same condition."))
                        elif primary_crop == hf_crop:
                            st.info(tr("Both models identify the same crop, but their disease predictions differ. Verify symptoms manually."))
                        else:
                            st.warning(tr("⚠️ **Secondary Disagreement:** The models predict different crops or conditions. Verify symptoms manually."))
                    elif hf_error and "503" in hf_error and "loading" in hf_error.lower():
                        st.warning(tr("Hugging Face model is loading. Please retry shortly."))
                    elif hf_error:
                        st.warning(f"{tr('Hugging Face cross-check failed')}: {hf_error}")
                        st.caption(tr("Check that the model ID is correct and enabled for Inference Providers, and that HF_API_TOKEN has Inference Providers permission."))
                    else:
                        st.caption(tr("Hugging Face returned an unexpected response. The primary ResNet50 prediction remains active."))
                else:
                    st.warning(tr("HF_API_TOKEN is missing. Add a Hugging Face token with Inference Providers permission to `.streamlit/secrets.toml`."))

                # Save to session history
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "disease": clean_name,
                    "raw_class": raw_class,
                    "confidence": confidence
                })

                # Downloadable report
                report_text = (
                    f"AgriSmart AI - Diagnosis Report\n"
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"{tr('Detected Condition')}: {display_name}\n"
                    f"Confidence: {confidence:.2f}%\n"
                    f"{tr('Severity')}: {tr(get_severity(raw_class))}\n\n"
                    f"{tr('📋 Recommended Action Plan')}:\n{treatment}\n"
                )
                report_generated_at = datetime.now()
                report_pdf = build_diagnosis_pdf(
                    condition=clean_name,
                    confidence=confidence,
                    severity=severity,
                    treatment=treatment,
                    generated_at=report_generated_at,
                )
                report_col_text, report_col_pdf = st.columns(2)
                with report_col_text:
                    st.download_button(
                        tr("⬇️️ Download Diagnosis Report (.txt)"),
                        data=report_text,
                        file_name=f"diagnosis_report_{report_generated_at.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                with report_col_pdf:
                    st.download_button(
                        tr("⬇️ Download PDF Diagnosis Report"),
                        data=report_pdf,
                        file_name=f"diagnosis_report_{report_generated_at.strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_diagnosis_report_pdf"
                    )
        else:
            st.info(tr("Upload a leaf image on the left to see diagnostic results here."))

    st.divider()
    st.subheader(tr("🤖 Agri Assistant"))
    st.caption(tr("Ask about crop diseases, spray practices, soil nutrients, or field conditions."))

    if "agri_chat_messages" not in st.session_state:
        st.session_state.agri_chat_messages = [{
            "role": "assistant",
            "content": tr("Hi! Tell me your crop and ask about symptoms, sprays, soil health, or weather."),
        }]

    for message in st.session_state.agri_chat_messages:
        with st.chat_message(message["role"]):
            content = tr(message["content"]) if message["role"] == "assistant" else message["content"]
            st.markdown(content)

    if question := st.chat_input(tr("Ask the Agri Assistant..."), key="agri_assistant_input"):
        st.session_state.agri_chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        answer = get_agri_assistant_response(question)
        st.session_state.agri_chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# ---------------------------------------------------------
# TAB 2: SOIL ADVISORY
# ---------------------------------------------------------
with tab2:
    st.subheader(tr("Soil Chemistry & Dosing Strategy"))
    st.caption(tr("Enter your soil lab test values to receive a customized nutrient advisory."))

    c1, c2, c3, c4 = st.columns(4)
    n_val = c1.number_input(tr("Nitrogen (N) - mg/kg"), min_value=0, max_value=300, value=140)
    p_val = c2.number_input(tr("Phosphorus (P) - mg/kg"), min_value=0, max_value=300, value=45)
    k_val = c3.number_input(tr("Potassium (K) - mg/kg"), min_value=0, max_value=300, value=180)
    ph_val = c4.number_input(tr("Soil pH Level"), min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    if st.button(tr("Compute Soil Nutrient Advisory"), type="primary"):
        st.markdown(tr("### 🧪 Soil Health Assessment"))

        advisories = []
        if n_val < 100:
            advisories.append(tr("• **Nitrogen Deficiency:** Apply Urea (46% N) at the recommended dosage to boost vegetative growth."))
        elif n_val > 200:
            advisories.append(tr("• **Excess Nitrogen:** Reduce nitrogenous fertilizer to avoid root burn and excessive leaf growth."))

        if p_val < 30:
            advisories.append(tr("• **Phosphorus Deficiency:** Apply Single Super Phosphate (SSP) to support root development."))
        if k_val < 150:
            advisories.append(tr("• **Potassium Deficiency:** Apply Muriate of Potash (MOP) to improve disease resistance."))

        if ph_val < 6.0:
            advisories.append(tr("• **Acidic Soil:** Apply agricultural limestone to raise the pH."))
        elif ph_val > 7.5:
            advisories.append(tr("• **Alkaline Soil:** Apply elemental sulfur or gypsum to lower the pH."))

        if not advisories:
            st.success(tr("✅ Soil chemical composition is optimal for general crop cultivation."))
        else:
            for item in advisories:
                st.warning(item)

    st.divider()
    st.subheader(tr("NPK Fertilizer Requirement Calculator"))
    st.caption(
        tr("Enter crop-specific nutrient targets. Rates are kg/acre; soil-test values above are mg/kg and are not converted directly. ")
        + tr("Marlas are converted at 160 marlas per acre. Calculations assume Urea 46-0-0, DAP 18-46-0, MOP 0-0-60, and 50 kg per bag.")
    )

    area_col, unit_col = st.columns([2, 1])
    area_unit = unit_col.selectbox(
        tr("Field area unit"),
        ["Acres", "Marlas"],
        format_func=tr,
        key="fertilizer_area_unit"
    )
    field_area = area_col.number_input(
        f"{tr('Field area')} ({tr(area_unit)})",
        min_value=0.1,
        max_value=100000.0,
        value=1.0,
        step=0.1 if area_unit == "Acres" else 1.0,
        key=f"fertilizer_area_{area_unit.lower()}"
    )

    rate_col_n, rate_col_p, rate_col_k = st.columns(3)
    n_rate = rate_col_n.number_input(
        tr("Target Nitrogen (N) - kg/acre"),
        min_value=0.0,
        max_value=500.0,
        value=60.0,
        step=1.0,
        key="fertilizer_n_rate"
    )
    p_rate = rate_col_p.number_input(
        tr("Target Phosphate (P₂O₅) - kg/acre"),
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
        key="fertilizer_p_rate"
    )
    k_rate = rate_col_k.number_input(
        tr("Target Potash (K₂O) - kg/acre"),
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
        key="fertilizer_k_rate"
    )

    area_acres = field_area if area_unit == "Acres" else field_area / 160
    required_dap = (p_rate * area_acres) / 0.46
    nitrogen_from_dap = required_dap * 0.18
    required_urea = max(0.0, (n_rate * area_acres - nitrogen_from_dap) / 0.46)
    required_mop = (k_rate * area_acres) / 0.60

    st.markdown(tr("#### Estimated Fertilizer Requirement"))
    fertilizer_columns = st.columns(3)
    fertilizer_results = [
        (tr("Urea"), required_urea),
        (tr("DAP"), required_dap),
        (tr("Potash (MOP)"), required_mop),
    ]
    for column, (fertilizer_name, quantity_kg) in zip(fertilizer_columns, fertilizer_results):
        full_bags = int(np.ceil(quantity_kg / 50)) if quantity_kg > 0 else 0
        bag_coverage = quantity_kg / (full_bags * 50) if full_bags else 0.0
        column.metric(
            fertilizer_name,
            f"{quantity_kg:.1f} kg",
            delta=f"{quantity_kg / 50:.2f} bags of 50 kg"
        )
        column.progress(
            bag_coverage,
            text=f"{full_bags} {tr('full bag' if full_bags == 1 else 'full bags')} {tr('to cover requirement')}"
        )

    st.caption(
        f"{tr('Calculated for')} {field_area:g} {tr(area_unit)}. "
        f"{tr('Urea is reduced by the nitrogen supplied through DAP. Confirm nutrient targets with local crop guidance.')}"
    )

# ---------------------------------------------------------
# TAB 3: WEATHER ENGINE
# ---------------------------------------------------------
with tab3:
    st.subheader(tr("Regional Weather Advisory"))
    st.caption(tr("Live weather data powered by OpenWeatherMap."))

    city = st.text_input(tr("Enter City / Region"), value="Sahiwal")

    if st.button(tr("Fetch Advisory")):
        api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
        if not api_key:
            st.error(tr("Weather API key not configured. Add OPENWEATHER_API_KEY in Streamlit secrets."))
        else:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            try:
                with st.spinner(f"{tr('Fetching live weather for')} {city}..."):
                    response = requests.get(url, timeout=8)
                    data = response.json()

                if response.status_code != 200:
                    st.error(f"{tr('Could not fetch weather for')} '{city}'. {data.get('message', '')}")
                else:
                    temp = data["main"]["temp"]
                    humidity = data["main"]["humidity"]
                    rain_prob = data.get("rain", {}).get("1h", 0)
                    raw_description = data["weather"][0]["description"].lower()
                    description = (
                        URDU_WEATHER_CONDITIONS.get(raw_description, "موسمی کیفیت")
                        if ui_language == "اردو"
                        else raw_description.title()
                    )

                    st.markdown(f"#### 📍 {tr('Regional Report')}: **{city}**")
                    metrics_column, spray_column = st.columns([3, 1.5], gap="medium")
                    with metrics_column:
                        with st.container(border=True):
                            w1, w2, w3 = st.columns(3)
                            w1.metric(tr("Temperature"), f"{temp:.1f} °C")
                            w2.metric(tr("Humidity"), f"{humidity}%")
                            w3.metric(tr("Rain (last 1h)"), f"{rain_prob} mm")
                            st.caption(f"{tr('Condition')}: {description}")

                    spray_status = get_spray_weather_status(temp, humidity, rain_prob)
                    with spray_column:
                        with st.container(border=True):
                            st.markdown(f"#### {tr('Smart Spray Weather Index')}")
                            if spray_status == "high_risk":
                                st.error(tr("⚠️ High Rain/Heat Risk - Avoid Spraying"))
                                st.caption(tr("Rainfall, high temperature, or very high humidity increases spray wash-off or crop-stress risk."))
                            elif spray_status == "favorable":
                                st.success(tr("✅ Favorable Spraying Window"))
                                st.caption(tr("Temperature, humidity, and recent rainfall are within the preferred range."))
                            else:
                                st.warning(tr("⚠️ Marginal Conditions - Recheck Before Spraying"))
                                st.caption(tr("Some weather values are outside the preferred range; recheck conditions and the product label."))
                            st.caption(tr("Wind speed is not included in this index. Avoid spraying in windy conditions and follow the product label."))

                    if spray_status == "high_risk":
                        st.warning(tr("🌧️ **Agronomic Insight:** Rain or heat risk detected. Delay chemical spraying to avoid wash-off or crop stress."))
                    elif spray_status == "favorable":
                        st.success(tr("🌦️ **Agronomic Insight:** Weather conditions are favorable for spraying. Low rainfall risk."))
                    else:
                        st.info(tr("**Agronomic Insight:** Conditions are mixed; check the product label and local wind before spraying."))

                    season, season_period, major_crops, winter_vegetables, moisture_note = get_weather_sowing_advice(
                        temp, humidity
                    )
                    st.markdown("---")
                    st.subheader(tr("Weather-Based Crop & Sowing Advisory"))
                    st.caption(
                        f"{tr('Current Season')}: {season} ({season_period}) · "
                        f"{temp:.1f} °C · {humidity}% {tr('Humidity')}"
                    )

                    major_column, vegetable_column = st.columns(2, gap="large")
                    with major_column:
                        st.markdown(f"#### {tr('Optimal Major Crops to Sow Now')}")
                        st.success("\n".join(f"- **{crop}**: {note}" for crop, note in major_crops))
                    with vegetable_column:
                        st.markdown(f"#### {tr('High-Value Winter Vegetables')}")
                        st.info("\n".join(f"- **{crop}**: {note}" for crop, note in winter_vegetables))
                        st.info(moisture_note)

                    st.markdown(f"#### {tr('Smog & Fungal Risk Alert')}")
                    alert_columns = st.columns(2)
                    smog_reported = any(
                        condition in raw_description
                        for condition in ("haze", "smoke", "dust", "fog", "mist", "sand", "ash")
                    )
                    with alert_columns[0]:
                        st.markdown(f"**{tr('Smog Alert')}**")
                        if smog_reported:
                            st.warning(tr("Smog, haze, or dust is reported. Reduce prolonged field work and protect seedlings from dust."))
                        else:
                            st.success(tr("No smog or dust condition is currently reported."))

                    with alert_columns[1]:
                        st.markdown(f"**{tr('Fungal Risk Alert')}**")
                        if humidity >= 80 or rain_prob > 0:
                            st.warning(tr("High humidity or rain may increase fungal pressure. Monitor leaves, avoid overhead irrigation, and maintain airflow."))
                        else:
                            st.info(tr("Humidity and rainfall are not elevated; maintain routine leaf monitoring and avoid overwatering."))

                    st.caption(tr(
                        "Temperature-based crop selection is a guide; verify planting dates, seed variety, and irrigation with local agricultural extension advice."
                    ))
            except requests.exceptions.RequestException:
                st.error(tr("Network error while fetching weather data. Check your internet connection."))