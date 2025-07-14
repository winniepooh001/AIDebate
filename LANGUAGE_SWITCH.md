# 语言切换说明 / Language Switch Guide

## 如何切换界面语言 / How to Switch UI Language

### 方法1：在界面中切换 / Method 1: Switch in UI
在侧边栏顶部的"界面语言 / UI Language"下拉菜单中选择：
- 中文：显示中文界面
- English：显示英文界面

In the sidebar, use the "界面语言 / UI Language" dropdown to select:
- 中文: Chinese interface
- English: English interface

### 方法2：代码中修改 / Method 2: Modify in Code

在 `debate_app.py` 文件中修改这一行：
```python
from src.ui_texts import ui_texts
```

改为：
```python
from src.ui_texts import UITexts
ui_texts = UITexts("english")  # 英文界面
# 或者
ui_texts = UITexts("chinese")  # 中文界面
```

To change in code, modify this line in `debate_app.py`:
```python
from src.ui_texts import ui_texts
```

To:
```python
from src.ui_texts import UITexts
ui_texts = UITexts("english")  # English interface
# or
ui_texts = UITexts("chinese")  # Chinese interface
```

### 方法3：环境变量 / Method 3: Environment Variable

可以设置环境变量 `UI_LANGUAGE`：
```bash
export UI_LANGUAGE=english  # 英文
export UI_LANGUAGE=chinese  # 中文
```

You can set the `UI_LANGUAGE` environment variable:
```bash
export UI_LANGUAGE=english  # English
export UI_LANGUAGE=chinese  # Chinese
```

## 支持的语言 / Supported Languages

- 中文 (chinese)
- English (english)

## 配置文件位置 / Configuration File Location

界面文字配置文件：`src/ui_texts.py`
UI text configuration file: `src/ui_texts.py`

## 添加新语言 / Adding New Languages

要添加新语言，请在 `UITexts` 类中：
1. 添加新的 `_load_[language]()` 方法
2. 在 `_load_texts()` 方法中添加相应的条件
3. 更新语言选择器

To add a new language, in the `UITexts` class:
1. Add a new `_load_[language]()` method
2. Add the corresponding condition in `_load_texts()` method
3. Update the language selector