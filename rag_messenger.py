import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTextEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QScrollArea, QMenuBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor, QKeySequence, QShortcut
from PyQt6.QtCore import pyqtSignal

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


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Messenger")
        self.setGeometry(100, 100, 800, 600)
        self._setup_ui()
        self._setup_menu()
        self._apply_styles()

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
        self.input_field.sendRequested.connect(self._send_message)  # Подключаем сигнал к методу
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("Введите сообщение...")
        self.send_btn = QPushButton("Отправить (Ctrl+Enter)")

        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._send_message)
        self.send_btn.clicked.connect(self._send_message)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field, 5)
        input_layout.addWidget(self.send_btn, 1)
        main_layout.addLayout(input_layout)

        self.input_field.setFocus()

    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        file_menu.addAction("Очистить чат", self._clear_chat)
        file_menu.addAction("Настройки", self._open_settings)
        file_menu.addSeparator()
        file_menu.addAction("Выход", self.close)

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

    def _send_message(self):
        if text := self.input_field.toPlainText().strip():
            self._add_message("Вы", text, USER_STYLE)
            self._add_message("RAG", "Вопрос принят.", BOT_STYLE)
            self.input_field.clear()
            self._scroll_to_bottom()

    def _add_message(self, sender, text, style):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        processed_text = text.replace('\n', '<br>')
        html = f'<div style="{style}"><b>{sender}:</b><br>{processed_text}<br></div>'
        cursor.insertHtml(html)
        self.chat_history.setTextCursor(cursor)

    def _clear_chat(self):
        self.chat_history.clear()

    def _open_settings(self):
        pass

    def _scroll_to_bottom(self):
        self.chat_history.ensureCursorVisible()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())