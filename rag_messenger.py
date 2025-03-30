import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTextEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QScrollArea, QDialog, QLabel, QComboBox, QDialogButtonBox,
                             QFileDialog, QMessageBox, QDoubleSpinBox, QSlider)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor, QKeySequence, QShortcut
import yaml
from rag_processor import *
import threading
from queue import Queue
from functools import wraps

USER_STYLE = """
    background: #DCF8C6;
    color: #000;
    border-radius: 10px;
    padding: 8px 12px;
    margin: 4px 0;
    max-width: 70%;
    float: left;
    clear: both;
"""

BOT_STYLE = """
    background: #E8E8E8;
    color: #000;
    border-radius: 10px;
    padding: 8px 12px;
    margin: 4px 0;
    max-width: 70%;
    float: right;
    clear: both;
"""


class PromptManager(QObject):
    prompts_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.prompts = {}
        self.current_prompt_name = ""
        self._load_prompts()

    def _load_prompts(self):
        """Загрузка промптов с автоматическим выбором первого"""
        try:
            with open("rag_prompts.yaml", "r", encoding="utf-8") as f:
                self.prompts = yaml.safe_load(f) or {}

                # Выбираем первый промпт независимо от имени
                if self.prompts:
                    self.current_prompt_name = next(iter(self.prompts.keys()))
                else:
                    self._create_fallback_prompts()

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            self._create_fallback_prompts()

    def _create_fallback_prompts(self):
        """Создание промптов по умолчанию при ошибке"""
        self.prompts = {
            "Автогенерированный промпт": {
                "system": "Ты - AI-ассистент. Отвечай на вопросы ясно и кратко.",
                "user": "Вопрос: {query}"
            }
        }
        self.current_prompt_name = next(iter(self.prompts.keys()))

    def get_current_prompt(self) -> tuple[str, str]:
        """Возвращает текущий промпт в виде (system, user)"""
        prompt = self.prompts.get(self.current_prompt_name, {})
        return prompt.get("system", ""), prompt.get("user", "")

    def get_prompt_names(self) -> list[str]:
        """Возвращает список доступных промптов"""
        return list(self.prompts.keys())

    def set_current_prompt(self, name: str):
        """Устанавливает текущий промпт"""
        if name in self.prompts:
            self.current_prompt_name = name
            self.prompts_updated.emit()

class CustomTextEdit(QTextEdit):
    sendRequested: pyqtSignal = pyqtSignal()  # Сигнал для отправки сообщения

    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.sendRequested.emit()  # Испускаем сигнал вместо вызова метода напрямую
            else:
                self.insertPlainText("\n")
        else:
            super().keyPressEvent(event)

class Dialog(RAGProcessor):
    def __init__(self, prompt_manager: PromptManager, llm, db, is_e5: bool):
        super().__init__()
        self.prompt_manager = prompt_manager
        self.db = db
        self.is_e5 = is_e5
        self.llm = llm
        self.temperature = 0.3
        self.summary = None
        self.consulter = DBConstructor()

    @staticmethod
    def threaded(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result_queue = Queue()

            def thread_worker():
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(result)
                except Exception as e:
                    result_queue.put(f"ERROR: {str(e)}")

            threading.Thread(target=thread_worker, daemon=True).start()
            return result_queue

        return wrapper

    @threaded
    def generate_answer(self, query: str, quest_hist: list):
        system, user_template = self.prompt_manager.get_current_prompt() # считал промпт в кортеж.
        found_chunks = self.db.similarity_search(f"query: {query}" if self.is_e5 else query, k=5) # Нашел 5 соответствующих отрезков
        context = ''.join([
            f"Фрагмент {n}:\n{chunk.page_content if not self.is_e5 else re.sub('passage: ', '', chunk.page_content)}\n"
            for n, chunk in enumerate(found_chunks)
        ]) # Собрал отрезки в строку
        if len(quest_hist) > 0:
            self.summary = f"Краткое содержание диалога: {self.summarizator([quest + ' ' + (ans if ans else None) for quest, ans in quest_hist])}"
            print(self.summary)
        user = user_template.format(summary=self.summary, query=query, context=context) # Собрал user по шаблону из саммари, вопроса и отрезков БЗ
        code, answer = self.consulter.request_to_openai(system, user, self.temperature, self.llm, True)

        if code: quest_hist.append((query, answer))

        return answer

    def summarizator(self, dialog: list):
        system = """Ты - третья сторона в диалоге. Твоя задача запомнить диалог и выделить из него самую суть.
        Если есть существенные детали, учти их."""
        user = f"Внимательно прочитай диалог, передай краткое содержание. Вот диалог: {' '.join(dialog)}. "

        print("Summarizator: ", user)
        code, summary = self.consulter.request_to_openai(system, user, 0, self.llm, True)
        return summary if code else None

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Messenger")
        self.setGeometry(100, 100, 800, 600)
        self.prompt_manager = PromptManager()
        self.prompt_manager.prompts_updated.connect(self._update_prompt)
        self._setup_ui()
        self._setup_menu()
        self._apply_styles()
        self.question_history = []
        # Инициализация базы (как было)
        self.consulter = DBConstructor()
        self.db = None  # Пока база не загружена
        self.is_e5 = False
        self.dialog = None
        self.current_emb_model = None
        self.current_mode = "Диалоговый"
        self.current_llm = "openai/gpt-4o-mini"
        self.current_database = "DB_Main_multilingual-e5-large"
        loaded = self._load_database(self.current_database)  # Переносим в отдельный метод
        if loaded: self._update_chat_headers()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # История чата
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setHtml("<div style='padding: 10px;'>")

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.chat_history)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # Поле ввода и кнопка
        self.input_field = CustomTextEdit()  # Используем кастомный виджет
        self.input_field.sendRequested.connect(self._send_user_message)  # Подключаем сигнал к методу
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("Введите сообщение...")
        self.send_btn = QPushButton("Отправить (Ctrl+Enter)")

        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._send_user_message)
        self.send_btn.clicked.connect(self._send_user_message)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field, 5)
        input_layout.addWidget(self.send_btn, 1)
        main_layout.addLayout(input_layout)

        self.input_field.setFocus()
        self._update_prompt()

    def _load_database(self, folder: str):
        """Загрузка/перезагрузка базы знаний"""
        self.data = self.consulter.faiss_loader(folder)
        s, meta = self.consulter.metadata_loader(folder)
        self.current_emb_model = meta['embedding_model']

        if self.data["success"]:
            self.db = self.data["db"]
            self.is_e5 = self.data["is_e5_model"]

            # Инициализация/переинициализация Dialog
            if self.dialog:
                self.dialog.db = self.db
                self.dialog.is_e5 = self.is_e5
            else:
                self.dialog = Dialog(self.prompt_manager, self.current_llm, self.db, self.is_e5)
                print(f"Текущая LLM заменена на {self.current_llm}")

            return True
        else:
            # QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки базы:\n{self.data['error']}")
            self._add_message("Ошибка", f"Ошибка загрузки базы:\n{self.data['error']}", BOT_STYLE)
            self.db = None
            self.dialog = None
            return False

    def load_new_database(self, folder: str):
        """Вызывается из SettingsDialog при смене базы"""
        self._load_database(folder)

    def _update_prompt(self):
        """Обновление текущего промпта"""
        system, user = self.prompt_manager.get_current_prompt()

    def _load_prompts(self):
        try:
            with open("rag_prompts.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Ошибка загрузки промптов: {e}")
            return None

    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        file_menu.addAction("Очистить чат", self._clear_chat)
        file_menu.addAction("Настройки", self._open_settings)
        file_menu.addSeparator()
        file_menu.addAction("Выход", self.close)

    def _open_settings(self):
        setts_dialog = SettingsDialog(self.prompt_manager, self.dialog, self)
        if setts_dialog.exec() == QDialog.DialogCode.Accepted:
            # Обновляем текущий промпт через менеджер
            self.prompt_manager.set_current_prompt(setts_dialog.prompt_combo.currentText())

            self.current_mode = setts_dialog.mode_combo.currentText()
            self.current_llm = setts_dialog.llm_combo.currentText()

            # Выводим новые настройки для отладки
            system, user = self.prompt_manager.get_current_prompt()
            print("\nОбновленные настройки:")
            print(f"• Режим: {self.current_mode}")
            print(f"• Текущий промпт: {self.prompt_manager.current_prompt_name}")
            print(f"• System prompt: {system[:50]}...")  # Выводим начало промпта для наглядности
            print(f"• User prompt: {user[:50]}...")
            print(f"• База FAISS: {self.current_emb_model}")
            print(f"• Модель генерации: {self.current_llm}\n")

            # Принудительное обновление интерфейса (опционально)
            self._update_chat_headers()

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI';
                font-size: 12pt;
            }
            QTextEdit {
                background: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton {
                background: #0088CC;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: #006699;
            }
        """)

    def _send_user_message(self):
        self.query = self.input_field.toPlainText().strip()
        if self.query:
            self._add_message("Вы", self.query, USER_STYLE)
            self.input_field.clear()
            self._scroll_to_bottom()

            # Запускаем генерацию через Dialog
            result_queue = self.dialog.generate_answer(self.query, self.question_history)
            self._handle_generation_result(result_queue)

    def _handle_generation_result(self, result_queue):
        def check_result():
            if result_queue.empty():
                QTimer.singleShot(100, check_result)
            else:
                answer = result_queue.get()

                if not answer.startswith("ERROR"):
                    self._add_message(f"RAG {self.current_llm.split('/')[-1]}", answer, BOT_STYLE)
                else:
                    self._add_message("Система", f"Ошибка генерации: {answer[7:]}", BOT_STYLE)

        check_result()

    def _add_message(self, sender, text, style):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        processed_text = text.replace('\n', '<br>')
        html = f'<div style="{style}"><b>{sender}:</b><br>{processed_text}<br></div>'
        cursor.insertHtml(html)
        self.chat_history.setTextCursor(cursor)

    def _clear_chat(self):
        self.chat_history.clear()

    def _scroll_to_bottom(self):
        self.chat_history.ensureCursorVisible()

    def _update_chat_headers(self):
        # Обновляем заголовки в истории чата
        self.chat_history.setHtml(f"""
                <div style="padding: 10px;">
                    <div style="text-align: left; margin-bottom: 20px;">
                        <b>Текущий режим:</b> {self.current_mode}<br>
                        <b>Промпт:</b> {self.prompt_manager.current_prompt_name}<br>
                        <b>Эмбеддинги:</b> {self.current_emb_model}<br>
                        <b>Генерация:</b> {self.current_llm}<br>
                        <b>Температура:</b> {self.dialog.temperature if self.dialog else 0.3}<br><br>
                    </div>
                </div>
        """)

class SettingsDialog(QDialog):
    def __init__(self, prompt_manager: PromptManager, dialog: Dialog, parent=None):
        super().__init__(parent)
        self.prompt_manager = prompt_manager
        self.dialog = dialog
        self.setWindowTitle("Настройки RAG")
        self.db_folder = None  # Будем хранить выбранную папку
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Режим работы
        layout.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Диалоговый", "Тестовый"])
        layout.addWidget(self.mode_combo)

        # Выбор промпта
        layout.addWidget(QLabel("Активный промпт:"))
        self.prompt_combo = QComboBox()
        self.prompt_combo.addItems(self.prompt_manager.get_prompt_names())
        self.prompt_combo.setCurrentText(self.prompt_manager.current_prompt_name)
        layout.addWidget(self.prompt_combo)

        # Выбор папки для базы знаний
        layout.addWidget(QLabel("База FAISS:"))

        self.db_btn = QPushButton("Выбрать папку с базой...")
        self.db_btn.clicked.connect(self._select_db_folder)
        layout.addWidget(self.db_btn)

        # Модели LLM
        layout.addWidget(QLabel("Модель генерации:"))
        self.llm_combo = QComboBox()
        self.llm_combo.addItems([
            "openai/gpt-4o-mini",
            "cohere/command-r7b-12-2024"
        ])
        layout.addWidget(self.llm_combo)

        # Добавляем элементы для температуры
        layout.addWidget(QLabel("Температура генерации:"))

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 10)
        self.temp_slider.setValue(3)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.3)

        # Связываем слайдер и спинбокс
        self.temp_slider.valueChanged.connect(
            lambda: self.temp_spin.setValue(self.temp_slider.value() / 10)
        )
        self.temp_spin.valueChanged.connect(
            lambda: self.temp_slider.setValue(int(self.temp_spin.value() * 10))
        )

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_spin)
        layout.addLayout(temp_layout)

        # Кнопки
        self.btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_box.accepted.connect(self._save_settings)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

    def _select_db_folder(self):
        """Выбор папки с базой FAISS"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с базой FAISS",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            # Проверка валидности папки
            if self._validate_db_folder(folder):
                self.db_folder = folder
                QMessageBox.information(self, "Успешно", "База успешно выбрана")
            else:
                QMessageBox.warning(self, "Ошибка", "Папка не содержит валидную базу FAISS")

    @staticmethod
    def _validate_db_folder(path: str) -> bool:
        """Проверка структуры папки с базой"""
        required = {'index.faiss', 'index.pkl', 'metadata.json'}
        files = set(os.listdir(path))
        return required.issubset(files)

    def _save_settings(self):
        """Сохранение выбранных настроек"""
        self.prompt_manager.set_current_prompt(self.prompt_combo.currentText())
        # Передаем выбранную папку в главное окно
        if self.db_folder:
            self.parent.load_new_database(self.db_folder)

        self.dialog.llm = self.llm_combo.currentText()
        self.dialog.temperature = self.temp_spin.value()

        self.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())