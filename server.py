from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import uvicorn
import uuid
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import aiohttp
from bs4 import BeautifulSoup
import base64
import re
import json

app = FastAPI(title="МАДИ University Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===
FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "qwen3-235b-a22b-fp8/latest")

if not FOLDER_ID or not API_KEY:
    raise ValueError("FOLDER_ID and API_KEY must be set in environment variables")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    project=FOLDER_ID
)

# Хранилище данных в памяти (для демо)
sessions = {}
orders = {}
chats = {}
files = {}
order_counter = 1
schedule_cache = {}
cache_expiry = {}

# Данные для МАДИ
MADI_DATA = {
    "groups": ["1МТ1", "1МТ2", "1МТ3", "1МТ4", "1МТ5", "2МТ1", "2МТ2", "2МТ3", "2МТ4", "3МТ1", "3МТ2", "3МТ3", "3МТ4"],
    "disciplines": [
        "Математический анализ",
        "Теоретическая механика",
        "Сопротивление материалов",
        "Информационная безопасность",
        "Программирование",
        "Базы данных",
        "Физика",
        "Химия",
        "Иностранный язык",
        "Экономика",
        "Менеджмент"
    ]
}

SYSTEM_PROMPT = """
Ты - университетский ассистент для студентов МАДИ. Твои задачи:
1. Помогать с вопросами о расписании занятий
2. Консультировать по учебным дисциплинам
3. Помогать с академическими вопросами
4. Объяснять сложные темы простым языком
5. Давать советы по учебе и организации времени

ВАЖНО: Отвечай дружелюбно и профессионально.
Используй обычный текст без Markdown форматирования.
Не выдумывай информацию о преподавателях или расписании.
"""

class OrderRequest(BaseModel):
    student_name: str
    discipline: str
    description: str
    course: Optional[str] = None
    variant: Optional[str] = None
    urgency: str
    group: str

def clean_ai_response(text):
    """Очищает текст от форматирования Markdown"""
    if not text:
        return ""
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()

# Парсинг расписания МАДИ
async def fetch_schedule(group: str):
    """Парсит расписание с сайта МАДИ для заданной группы"""
    global schedule_cache, cache_expiry
    
    # Проверяем кэш
    now = datetime.now()
    if group in schedule_cache and group in cache_expiry and now < cache_expiry[group]:
        return schedule_cache[group]
    
    try:
        # URL для парсинга расписания МАДИ
        url = f"https://raspisanie.madi.ru/tplan/r/?task=7&group={group}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Парсим расписание (упрощенная версия)
                    schedule = parse_schedule_from_html(soup, group)
                    
                    # Кэшируем на 1 час
                    schedule_cache[group] = schedule
                    cache_expiry[group] = now + timedelta(hours=1)
                    
                    return schedule
                else:
                    print(f"Ошибка HTTP: {response.status}")
                    return get_mock_schedule(group)
                    
    except Exception as e:
        print(f"Ошибка парсинга расписания: {e}")
        return get_mock_schedule(group)

def parse_schedule_from_html(soup, group):
    """Парсит расписание из HTML сайта МАДИ"""
    # Это упрощенный парсер - в реальности нужно адаптировать под структуру сайта
    try:
        # Ищем таблицу с расписанием
        schedule_tables = soup.find_all('table', class_='schedule')
        
        if not schedule_tables:
            return get_mock_schedule(group)
            
        # Здесь должна быть логика парсинга реального расписания
        # Возвращаем мок-данные для демонстрации
        return get_mock_schedule(group)
        
    except Exception as e:
        print(f"Ошибка парсинга HTML: {e}")
        return get_mock_schedule(group)

def get_mock_schedule(group):
    """Возвращает мок-расписание для демонстрации"""
    return {
        "числитель": [
            {
                "day": "Понедельник",
                "date": datetime.now().strftime("%d.%m"),
                "lessons": [
                    {"time": "9:00-10:30", "subject": "Математический анализ", "room": "101", "teacher": "Иванов А.П.", "type": "лекция"},
                    {"time": "10:45-12:15", "subject": "Физика", "room": "203", "teacher": "Петрова С.М.", "type": "практика"},
                    {"time": "13:00-14:30", "subject": "Информационная безопасность", "room": "305", "teacher": "Сидоров В.К.", "type": "лабораторная"}
                ]
            },
            {
                "day": "Вторник",
                "date": (datetime.now() + timedelta(days=1)).strftime("%d.%m"),
                "lessons": [
                    {"time": "9:00-10:30", "subject": "Программирование", "room": "305", "teacher": "Сидоров В.К.", "type": "лекция"},
                    {"time": "10:45-12:15", "subject": "Теоретическая механика", "room": "210", "teacher": "Николаев Д.С.", "type": "практика"}
                ]
            }
        ],
        "знаменатель": [
            {
                "day": "Понедельник",
                "date": datetime.now().strftime("%d.%m"),
                "lessons": [
                    {"time": "9:00-10:30", "subject": "Иностранный язык", "room": "105", "teacher": "Козлова Е.В.", "type": "практика"},
                    {"time": "10:45-12:15", "subject": "Сопротивление материалов", "room": "210", "teacher": "Николаев Д.С.", "type": "лекция"}
                ]
            },
            {
                "day": "Вторник",
                "date": (datetime.now() + timedelta(days=1)).strftime("%d.%m"),
                "lessons": [
                    {"time": "9:00-10:30", "subject": "Базы данных", "room": "401", "teacher": "Федоров М.П.", "type": "лабораторная"},
                    {"time": "10:45-12:15", "subject": "Экономика", "room": "108", "teacher": "Васильева О.Н.", "type": "лекция"}
                ]
            }
        ]
    }

def calculate_lesson_progress(lesson_time):
    """Рассчитывает прогресс текущей пары"""
    try:
        start_str, end_str = lesson_time.split('-')
        start_time = datetime.strptime(start_str, '%H:%M').time()
        end_time = datetime.strptime(end_str, '%H:%M').time()
        now_time = datetime.now().time()
        
        total_duration = (datetime.combine(datetime.today(), end_time) -
                         datetime.combine(datetime.today(), start_time)).seconds
        elapsed = (datetime.combine(datetime.today(), now_time) -
                  datetime.combine(datetime.today(), start_time)).seconds
        
        if elapsed < 0:
            return 0  # Пара еще не началась
        elif elapsed > total_duration:
            return 100  # Пара уже закончилась
        else:
            return min(100, int((elapsed / total_duration) * 100))
            
    except:
        return 0

# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("university_assistant.html")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL}

# API endpoints
@app.get("/api/groups")
async def get_groups():
    return MADI_DATA["groups"]

@app.get("/api/disciplines")
async def get_disciplines():
    return MADI_DATA["disciplines"]

@app.get("/api/schedule/{group}")
async def get_schedule(group: str):
    schedule = await fetch_schedule(group)
    if schedule:
        # Добавляем прогресс для текущих пар
        for week_type in ["числитель", "знаменатель"]:
            for day in schedule.get(week_type, []):
                for lesson in day.get("lessons", []):
                    lesson["progress"] = calculate_lesson_progress(lesson["time"])
        return schedule
    raise HTTPException(status_code=404, detail="Расписание не найдено")

@app.post("/api/order")
async def create_order(request: Request):
    global order_counter
    try:
        data = await request.json()
        
        order_id = order_counter
        order_counter += 1
        
        # Создаем заказ
        orders[order_id] = {
            "order_id": order_id,
            "student_name": data.get("student_name"),
            "discipline": data.get("discipline"),
            "description": data.get("description"),
            "course": data.get("course"),
            "variant": data.get("variant"),
            "urgency": data.get("urgency"),
            "group": data.get("group"),
            "status": "ожидает оценки",
            "price": None,
            "created_at": datetime.now().isoformat(),
            "manager_message": "Ваша заявка принята в обработку. Менеджер свяжется с вами в течение 15 минут."
        }
        
        # Инициализируем чат для этого заказа
        chats[order_id] = [{
            "sender": "system",
            "text": "Заявка создана. Ожидайте ответа менеджера.",
            "timestamp": datetime.now().isoformat(),
            "files": []
        }]
        
        return orders[order_id]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/orders")
async def get_orders():
    return list(orders.values())

@app.get("/api/order/{order_id}")
async def get_order(order_id: int):
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Заказ не найден")
    return orders[order_id]

@app.get("/api/chat/{order_id}")
async def get_chat(order_id: int):
    if order_id not in chats:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return chats[order_id]

@app.post("/api/chat/{order_id}")
async def send_message(order_id: int, request: Request):
    if order_id not in chats:
        raise HTTPException(status_code=404, detail="Чат не найден")
    
    try:
        data = await request.json()
        text = data.get("text", "").strip()
        sender = data.get("sender", "student")
        file_names = data.get("files", [])
        
        if not text and not file_names:
            raise HTTPException(status_code=400, detail="Пустое сообщение")
        
        chat_message = {
            "sender": sender,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "files": file_names
        }
        
        chats[order_id].append(chat_message)
        
        return {"status": "ok", "message": chat_message}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/upload/{order_id}")
async def upload_file(order_id: int, file: UploadFile = File(...)):
    if order_id not in chats:
        raise HTTPException(status_code=404, detail="Чат не найден")
    
    try:
        # Читаем файл
        content = await file.read()
        
        # Проверяем размер файла (100 МБ)
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Файл слишком большой (максимум 100 МБ)")
        
        # Сохраняем файл
        if order_id not in files:
            files[order_id] = {}
        
        file_id = str(uuid.uuid4())
        files[order_id][file_id] = {
            "filename": file.filename,
            "content": content,
            "content_type": file.content_type,
            "uploaded_at": datetime.now().isoformat()
        }
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "message": "Файл успешно загружен"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки файла: {str(e)}")

@app.get("/api/file/{order_id}/{file_id}")
async def download_file(order_id: int, file_id: str):
    if order_id not in files or file_id not in files[order_id]:
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    file_data = files[order_id][file_id]
    
    return JSONResponse(
        content={
            "filename": file_data["filename"],
            "content": base64.b64encode(file_data["content"]).decode('utf-8'),
            "content_type": file_data["content_type"]
        }
    )

@app.post("/api/order/{order_id}/set_price")
async def set_order_price(order_id: int, request: Request):
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Заказ не найден")
    
    try:
        data = await request.json()
        price = data.get("price")
        
        if not price:
            raise HTTPException(status_code=400, detail="Цена не указана")
        
        orders[order_id]["price"] = price
        orders[order_id]["status"] = "ожидает оплаты"
        
        # Добавляем сообщение в чат
        if order_id in chats:
            chat_message = {
                "sender": "manager",
                "text": f"Стоимость работы: {price}₽. Для начала выполнения необходимо произвести оплату.",
                "timestamp": datetime.now().isoformat(),
                "files": []
            }
            chats[order_id].append(chat_message)
        
        return {"status": "ok", "price": price}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/order/{order_id}/pay")
async def pay_order(order_id: int):
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Заказ не найден")
    
    # Имитируем успешную оплату
    orders[order_id]["status"] = "оплачено"
    orders[order_id]["paid_at"] = datetime.now().isoformat()
    
    # Добавляем сообщение в чат
    if order_id in chats:
        chat_message = {
            "sender": "system",
            "text": "✅ Оплата прошла успешно! Менеджер приступает к выполнению работы.",
            "timestamp": datetime.now().isoformat(),
            "files": []
        }
        chats[order_id].append(chat_message)
    
    return {"status": "оплачено", "message": "Оплата прошла успешно"}

@app.post("/api/order/{order_id}/complete")
async def complete_order(order_id: int):
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Заказ не найден")
    
    orders[order_id]["status"] = "завершено"
    orders[order_id]["completed_at"] = datetime.now().isoformat()
    
    # Добавляем сообщение в чат
    if order_id in chats:
        chat_message = {
            "sender": "manager",
            "text": "✅ Работа завершена и отправлена вам. Спасибо за заказ!",
            "timestamp": datetime.now().isoformat(),
            "files": []
        }
        chats[order_id].append(chat_message)
    
    return {"status": "завершено", "message": "Заказ завершен"}

# AI Chat endpoint
@app.post("/api/ai/chat")
async def chat_with_ai(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id", str(uuid.uuid4()))

        if not message:
            return JSONResponse(status_code=400, content={"error": "empty message"})

        if session_id not in sessions:
            sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

        sessions[session_id].append({"role": "user", "content": message})

        response = client.responses.create(
            model=f"gpt://{FOLDER_ID}/{MODEL}",
            temperature=0.6,
            max_output_tokens=2500,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": message}]
        )

        reply = response.output_text.strip()
        cleaned_reply = clean_ai_response(reply)

        sessions[session_id].append({"role": "assistant", "content": cleaned_reply})

        return {"reply": cleaned_reply, "session_id": session_id}

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("ОШИБКА В AI ЧАТЕ:")
        print(error_detail)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Ошибка соединения с AI-сервисом"}
        )

# Admin endpoints
@app.get("/api/admin/orders")
async def get_all_orders():
    """Получить все заказы для админ-панели"""
    return {
        "orders": list(orders.values()),
        "total": len(orders),
        "pending": len([o for o in orders.values() if o["status"] == "ожидает оценки"]),
        "waiting_payment": len([o for o in orders.values() if o["status"] == "ожидает оплаты"]),
        "paid": len([o for o in orders.values() if o["status"] == "оплачено"]),
        "completed": len([o for o in orders.values() if o["status"] == "завершено"])
    }

@app.get("/api/admin/order/{order_id}/full")
async def get_full_order_info(order_id: int):
    """Полная информация о заказе для админки"""
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Заказ не найден")
    
    order_info = orders[order_id].copy()
    order_info["chat"] = chats.get(order_id, [])
    order_info["files"] = list(files.get(order_id, {}).keys())
    
    return order_info

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
