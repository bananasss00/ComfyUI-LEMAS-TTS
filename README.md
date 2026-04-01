# ComfyUI LEMAS-TTS

[English](#english) | [Русский](#русский)

---

<a name="english"></a>
# 🇬🇧 ComfyUI LEMAS-TTS (English)

ComfyUI custom nodes for **[LEMAS-TTS](https://github.com/LEMAS-Project/LEMAS-TTS)** — a state-of-the-art, zero-shot, multilingual Text-to-Speech model with advanced prosody and emotion transfer capabilities.

### ✨ Features
* **Zero-Shot Voice Cloning:** Clone any voice using just a 3-10 second audio sample.
* **Multilingual Support:** Seamlessly generate speech in multiple languages.
* **Prosody Transfer:** Capture the exact emotion, intonation, and rhythm of the reference audio.
* **Built-in UVR5 Denoiser:** Clean your reference audio directly inside ComfyUI for better cloning results.
* **Fully Automated Setup:** Missing Python packages and HuggingFace model weights are downloaded automatically on the first run.
* **Optimized UI:** Clean node design with helpful tooltips for every parameter.

### 📥 Installation
1. Navigate to your ComfyUI `custom_nodes` directory.
2. Clone this repository:
```bash
git clone https://github.com/bananasss00/ComfyUI-LEMAS-TTS.git
```
3. Restart ComfyUI. 
*Note: On the first run, the node will automatically install required pip packages and download the model weights (~few GBs) into `ComfyUI/models/lemas_tts`.*

### 🧩 Nodes Overview

* **LEMAS TTS Model Loader:**
  * Select between `multilingual_grl` (stable baseline) and `multilingual_prosody` (highly expressive, transfers emotions).
* **LEMAS TTS Denoise (UVR5):**
  * Takes your raw reference audio and removes background noise/music. Clean audio = clean voice clone.
* **LEMAS TTS Synthesize:**
  * The core node. Provide the reference audio, its exact transcript (`ref_text`), and the text you want to generate (`gen_text`).

### 💡 Tips for Perfect Voice Cloning (Fixing "Hallucinations" & "Leakage")
If the generated audio contains words from the reference audio, mumbles, or sounds distorted, follow these rules:
1. **Exact Transcription:** Your `ref_text` must match the audio sample *word for word*. If the speaker stutters or says "umm", either transcribe it or crop it out of the audio.
2. **ALWAYS use a period:** Make sure your `ref_text` ends with a period (`.`) or an exclamation mark (`!`). This tells the model the reference phrase is complete.
3. **Crop the silence:** The reference audio should start exactly when the speech starts and end exactly when it finishes. Trailing silence confuses the model.
4. **Increase CFG Strength:** If the model hallucinates or ignores your text, increase `cfg_strength` from `5.0` to `5.5` or `6.0`.
5. **Use Newlines:** For long texts, separate sentences with a new line (Enter). The node will process them sequentially and stitch them together, preventing context loss.

---

<a name="русский"></a>
# 🇷🇺 ComfyUI LEMAS-TTS (Русский)

Пользовательские ноды ComfyUI для интеграции **[LEMAS-TTS](https://github.com/LEMAS-Project/LEMAS-TTS)** — передовой мультиязычной нейросети для синтеза речи (Zero-Shot TTS) с продвинутым переносом интонаций и эмоций.

### ✨ Особенности
* **Zero-Shot клонирование голоса:** Клонируйте любой голос, используя аудио-образец длиной всего 3-10 секунд.
* **Мультиязычность:** Модель отлично справляется с разными языками (в т.ч. русским).
* **Перенос просодии:** Точное копирование эмоций, ритма и тембра оригинального спикера.
* **Встроенный шумодав UVR5:** Очищайте референсное аудио от фонового шума прямо внутри ComfyUI.
* **Полная автоматизация:** Недостающие Python-библиотеки и веса моделей скачиваются автоматически при первом запуске.
* **Удобный интерфейс:** Ноды очищены от лишнего, ко всем параметрам добавлены подробные подсказки (tooltips) на русском и английском.

### 📥 Установка
1. Перейдите в папку `custom_nodes` вашего ComfyUI.
2. Склонируйте этот репозиторий:
```bash
git clone https://github.com/bananasss00/ComfyUI-LEMAS-TTS.git
```
3. Перезапустите ComfyUI.
*Примечание: При первом запуске генерации нода автоматически установит нужные библиотеки и скачает веса моделей (несколько ГБ) в папку `ComfyUI/models/lemas_tts`. Это может занять время.*

### 🧩 Описание нод

* **LEMAS TTS Model Loader:**
  * Выбор архитектуры: `multilingual_grl` (базовая, стабильная) или `multilingual_prosody` (максимально выразительная, переносит эмоции).
* **LEMAS TTS Denoise (UVR5):**
  * Удаляет фоновый шум из аудио-образца. Чистый референс — залог качественного клонирования.
* **LEMAS TTS Synthesize:**
  * Главная нода. Принимает аудио-образец, его точную расшифровку (`ref_text`) и текст, который нужно озвучить (`gen_text`).

### 💡 Советы для идеального клонирования (Как исправить "бормотание" и "утечки")
Если в генерацию пролезают куски фраз из референса, голос бормочет или искажается, соблюдайте эти 5 правил:
1. **Точная расшифровка:** Поле `ref_text` должно совпадать с аудио-образцом *слово в слово*. Если человек в аудио запинается или "экает" — либо напишите это в тексте, либо отрежьте этот кусок в аудиоредакторе.
2. **ОБЯЗАТЕЛЬНО ставьте точку:** Ваш `ref_text` **обязан** заканчиваться точкой (`.`) или восклицательным знаком (`!`). Это дает модели команду: "старая фраза закончилась, начинай новую с чистого листа".
3. **Обрезайте тишину:** Аудио-образец должен начинаться и заканчиваться ровно со словами спикера. Висящая в конце тишина или вздохи сводят модель с ума.
4. **Повысьте CFG Strength:** Если модель игнорирует ваш текст и фантазирует, увеличьте параметр `cfg_strength` с `5.0` до `5.5` или `6.0`.
5. **Разделяйте длинный текст:** Пишите длинный текст с переносами строк (Enter). Нода будет генерировать каждую строку отдельно, а затем склеит их. Это не даст модели "потерять контекст" на длинных дистанциях.

---

### 📝 Credits / Благодарности
* Based on the official [LEMAS-TTS repository](https://github.com/LEMAS-Project/LEMAS-TTS).
