from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import uvicorn
from transformers import pipeline

app = FastAPI(title="Модерация комментариев")

app.mount("/static", StaticFiles(directory="static"), name="static")

print("Загрузка модели ruBERT-tiny-toxicity...")
classifier = pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity")
print("Модель загружена!")

def check_toxicity(text: str):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    
    toxic_labels = ['insult', 'toxicity', 'profanity', 'toxic', 'obscene']
    
    safe_labels = ['non-toxic', 'neutral', 'positive', 'clean']
    
    if label in toxic_labels:
        is_toxic = True
    elif label in safe_labels:
        is_toxic = False
    else:
        is_toxic = score > 0.7
    
    return is_toxic, score

def simple_email_validate(email: str) -> bool:
    if len(email) < 5:
        return False
    if '@' not in email:
        return False
    local, domain = email.split('@', 1)
    if len(local) == 0 or len(domain) < 3:
        return False
    if '.' not in domain:
        return False
    return True

class FeedbackRequest(BaseModel):
    email: str
    subject: str
    text: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if not simple_email_validate(v):
            raise ValueError('Введите корректный email (например, name@mail.ru)')
        return v

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Текст сообщения не может быть пустым')
        return v.strip()

    @field_validator('subject')
    @classmethod
    def subject_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Тема обращения не может быть пустой')
        return v.strip()

class TextRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Текст не может быть пустым')
        return v.strip()

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    found_word: str | None = None

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTML_CONTENT

@app.post("/predict")
async def predict(request: TextRequest):
    is_toxic, score = check_toxicity(request.text)
    
    if is_toxic:
        return PredictionResponse(
            text=request.text,
            prediction="TOXIC",
            confidence=score,
            found_word=None
        )
    else:
        return PredictionResponse(
            text=request.text,
            prediction="NON-TOXIC",
            confidence=score,
            found_word=None
        )

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    is_toxic, score = check_toxicity(request.text)
    
    if is_toxic:
        return PredictionResponse(
            text=request.text,
            prediction="TOXIC",
            confidence=score,
            found_word=None
        )
    else:
        return {
            "status": "success",
            "message": "Обращение отправлено",
            "email": request.email,
            "subject": request.subject,
            "text": request.text
        }

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Система модерации | ГАПОУ "ОКЭИ"</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', 'Roboto', system-ui, sans-serif;
            background: #f0f2f5;
            min-height: 100vh;
            padding: 40px 24px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            margin-bottom: 32px;
            padding-bottom: 20px;
            border-bottom: 2px solid #d0d5dd;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .logo {
            flex-shrink: 0;
            width: 120px;
            height: 80px;
            background: #1e3a5f;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .logo img { width: 100%; height: 100%; object-fit: cover; }
        .header-text { flex: 1; }
        .header-text h1 { font-size: 28px; font-weight: 600; color: #1a2c3e; margin-bottom: 8px; }
        .header-text p { font-size: 14px; color: #5a6874; }
        .header-text .institution { font-size: 12px; color: #6c7a8a; margin-top: 8px; }
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 28px;
            background: #ffffff;
            padding: 4px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }
        .tab {
            flex: 1;
            background: transparent;
            border: none;
            padding: 12px 16px;
            font-size: 14px;
            font-weight: 500;
            color: #4a5a6e;
            cursor: pointer;
            border-radius: 10px;
            transition: all 0.2s;
        }
        .tab:hover { background: #f1f5f9; color: #1e3a5f; }
        .tab.active { background: #1e3a5f; color: white; }
        .card {
            background: #ffffff;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 32px;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.2s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .form-group { margin-bottom: 24px; }
        label { display: block; font-size: 13px; font-weight: 500; color: #2c3e50; margin-bottom: 6px; }
        input, textarea, select {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            font-size: 14px;
            font-family: inherit;
            background: #ffffff;
            transition: 0.2s;
        }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #2c6e9e;
            box-shadow: 0 0 0 2px rgba(44,110,158,0.1);
        }
        textarea { resize: vertical; min-height: 110px; }
        .row { display: flex; gap: 20px; margin-bottom: 24px; }
        .row .form-group { flex: 1; margin-bottom: 0; }
        .btn {
            width: 100%;
            background: #1e3a5f;
            border: none;
            padding: 12px 20px;
            border-radius: 40px;
            font-size: 14px;
            font-weight: 500;
            color: white;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 8px;
        }
        .btn:hover { background: #0f2b47; }
        .result-card {
            margin-top: 28px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 16px;
            border-left: 4px solid #3b7a9e;
        }
        .result-card h4 { font-size: 12px; font-weight: 600; text-transform: uppercase; color: #5a6e82; margin-bottom: 8px; }
        .result-card p { font-size: 14px; color: #1e2a3a; word-break: break-word; }
        .status-badge { display: inline-block; padding: 6px 14px; border-radius: 30px; font-size: 12px; font-weight: 500; margin-top: 12px; }
        .status-pending { background: #fef9e6; color: #b65f00; }
        .status-approved { background: #e6f4ea; color: #1f7840; }
        .status-rejected { background: #feeceb; color: #bc3f2e; }
        .info-block {
            background: #f4f7fc;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 28px;
            font-size: 13px;
            color: #2c3e50;
            border-left: 3px solid #2c6e9e;
        }
        .error-message { color: #bc3f2e; font-size: 12px; margin-top: 5px; display: none; }
        .footer { margin-top: 32px; text-align: center; font-size: 12px; color: #8c9aac; border-top: 1px solid #e2e8f0; padding-top: 24px; }
        @media (max-width: 640px) {
            .header { flex-direction: column; text-align: center; }
            .tabs { flex-direction: column; border-radius: 12px; }
            .card { padding: 20px; }
            .row { flex-direction: column; gap: 0; }
            .header-text h1 { font-size: 24px; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="logo"><img src="/static/logo.png" alt="Логотип ОКЭИ"></div>
        <div class="header-text">
            <h1>Система модерации сообщений</h1>
            <p>Автоматическая проверка текстовых обращений на соответствие правилам</p>
            <div class="institution">ГАПОУ «Оренбургский колледж экономики и информатики»</div>
        </div>
    </div>
    <div class="tabs">
        <button class="tab active" data-tab="feedback">Обратная связь</button>
        <button class="tab" data-tab="forum">Форум / Чат</button>
        <button class="tab" data-tab="application">Заявления и анкеты</button>
    </div>
    <div class="card">
        <div class="tab-content active" id="feedback">
            <div class="info-block">Форма обратной связи предназначена для направления вопросов, предложений и отзывов о работе колледжа.</div>
            <div class="form-group"><label>Электронная почта *</label><input type="email" id="feedback-email" placeholder="ivanov@okey.ru"><div class="error-message" id="feedback-email-error">Введите корректный email (например, name@okey.ru)</div></div>
            <div class="form-group"><label>Тема обращения *</label><input type="text" id="feedback-subject" placeholder="Вопрос о расписании"><div class="error-message" id="feedback-subject-error">Тема не может быть пустой</div></div>
            <div class="form-group"><label>Текст сообщения *</label><textarea id="feedback-text" placeholder="Введите текст сообщения..."></textarea><div class="error-message" id="feedback-text-error">Текст не может быть пустым</div></div>
            <button class="btn" onclick="sendFeedback()">Отправить сообщение</button>
            <div class="result-card" id="feedback-result" style="display: none;"><h4>Результат проверки</h4><p id="feedback-result-text">-</p><div id="feedback-status" class="status-badge status-pending">Ожидание проверки</div></div>
        </div>
        <div class="tab-content" id="forum">
            <div class="info-block">Учебный форум и чат — обсуждение лекций, помощь в выполнении заданий.</div>
            <div class="form-group"><label>ФИО или псевдоним</label><input type="text" id="forum-name" placeholder="Студент группы 3пк2"></div>
            <div class="form-group"><label>Раздел обсуждения</label><select id="forum-topic"><option>Лекция</option><option>Практическое занятие</option><option>Домашнее задание</option><option>Общий чат</option></select></div>
            <div class="form-group"><label>Текст комментария *</label><textarea id="forum-text" placeholder="Напишите комментарий..."></textarea></div>
            <button class="btn" onclick="sendMessage('forum')">Опубликовать</button>
            <div class="result-card" id="forum-result" style="display: none;"><h4>Результат проверки</h4><p id="forum-result-text">-</p><div id="forum-status" class="status-badge status-pending">Ожидание проверки</div></div>
        </div>
        <div class="tab-content" id="application">
            <div class="info-block">Подача заявлений, анкет и официальных обращений.</div>
            <div class="row"><div class="form-group"><label>Фамилия</label><input type="text" id="app-lastname" placeholder="Иванов"></div><div class="form-group"><label>Имя</label><input type="text" id="app-firstname" placeholder="Иван"></div></div>
            <div class="form-group"><label>Группа / Должность</label><input type="text" id="app-group" placeholder="3пк2"></div>
            <div class="form-group"><label>Тип документа</label><select id="app-type"><option>Заявление на перевод</option><option>Заявление на стипендию</option><option>Академический отпуск</option><option>Обращение к директору</option></select></div>
            <div class="form-group"><label>Текст заявления *</label><textarea id="app-text" placeholder="Изложите суть обращения..."></textarea></div>
            <button class="btn" onclick="sendMessage('application')">Отправить заявление</button>
            <div class="result-card" id="application-result" style="display: none;"><h4>Результат проверки</h4><p id="application-result-text">-</p><div id="application-status" class="status-badge status-pending">Ожидание проверки</div></div>
        </div>
    </div>
    <div class="footer"><p>ГАПОУ «Оренбургский колледж экономики и информатики» · Система автоматической модерации сообщений</p></div>
</div>
<script>
    document.querySelectorAll('.tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.getAttribute('data-tab')).classList.add('active');
            clearResults();
        });
    });
    function clearResults() {
        ['feedback', 'forum', 'application'].forEach(id => {
            const el = document.getElementById(id + '-result');
            if (el) el.style.display = 'none';
        });
    }
    function validateEmail(email) {
        if (email.length < 5) return false;
        if (email.indexOf('@') === -1) return false;
        let parts = email.split('@');
        if (parts.length !== 2) return false;
        if (parts[0].length === 0) return false;
        if (parts[1].length < 3) return false;
        if (parts[1].indexOf('.') === -1) return false;
        return true;
    }
    function hideErrors() {
        ['feedback-email-error', 'feedback-subject-error', 'feedback-text-error'].forEach(id => {
            document.getElementById(id).style.display = 'none';
        });
    }
    function validateFeedbackForm() {
        let isValid = true;
        const email = document.getElementById('feedback-email').value;
        const subject = document.getElementById('feedback-subject').value;
        const text = document.getElementById('feedback-text').value;
        hideErrors();
        if (!validateEmail(email)) {
            document.getElementById('feedback-email-error').style.display = 'block';
            isValid = false;
        }
        if (!subject.trim()) {
            document.getElementById('feedback-subject-error').style.display = 'block';
            isValid = false;
        }
        if (!text.trim()) {
            document.getElementById('feedback-text-error').style.display = 'block';
            isValid = false;
        }
        return isValid;
    }
    async function sendFeedback() {
        if (!validateFeedbackForm()) return;
        const email = document.getElementById('feedback-email').value;
        const subject = document.getElementById('feedback-subject').value;
        const text = document.getElementById('feedback-text').value;
        const resultDiv = document.getElementById('feedback-result');
        const resultText = document.getElementById('feedback-result-text');
        const statusDiv = document.getElementById('feedback-status');
        resultDiv.style.display = 'block';
        resultText.textContent = `«${text.substring(0, 120)}${text.length > 120 ? '…' : ''}»`;
        statusDiv.textContent = 'Проверка...';
        statusDiv.className = 'status-badge status-pending';
        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, subject, text })
            });
            const data = await response.json();
            if (data.prediction === 'TOXIC') {
                statusDiv.textContent = `Отклонено: сообщение содержит токсичный контент (уверенность ${Math.round(data.confidence * 100)}%)`;
                statusDiv.className = 'status-badge status-rejected';
            } else if (data.status === 'success') {
                statusDiv.textContent = '✅ Сообщение успешно отправлено!';
                statusDiv.className = 'status-badge status-approved';
                document.getElementById('feedback-email').value = '';
                document.getElementById('feedback-subject').value = '';
                document.getElementById('feedback-text').value = '';
            } else {
                statusDiv.textContent = 'Ошибка при отправке';
                statusDiv.className = 'status-badge status-rejected';
            }
        } catch (error) {
            statusDiv.textContent = 'Ошибка подключения к серверу';
            statusDiv.className = 'status-badge status-rejected';
        }
    }
    async function sendMessage(type) {
        let text = type === 'forum' ? document.getElementById('forum-text').value : document.getElementById('app-text').value;
        if (!text.trim()) { alert('Пожалуйста, заполните текст сообщения'); return; }
        const resultDiv = document.getElementById(type + '-result');
        const resultText = document.getElementById(type + '-result-text');
        const statusDiv = document.getElementById(type + '-status');
        resultDiv.style.display = 'block';
        resultText.textContent = `«${text.substring(0, 120)}${text.length > 120 ? '…' : ''}»`;
        statusDiv.textContent = 'Проверка...';
        statusDiv.className = 'status-badge status-pending';
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await response.json();
            if (data.prediction === 'TOXIC') {
                statusDiv.textContent = `Отклонено: сообщение содержит токсичный контент (уверенность ${Math.round(data.confidence * 100)}%)`;
                statusDiv.className = 'status-badge status-rejected';
            } else {
                statusDiv.textContent = 'Одобрено: сообщение соответствует установленным требованиям';
                statusDiv.className = 'status-badge status-approved';
            }
        } catch (error) {
            statusDiv.textContent = 'Ошибка подключения к серверу модерации';
            statusDiv.className = 'status-badge status-rejected';
        }
    }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)