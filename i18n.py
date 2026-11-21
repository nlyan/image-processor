"""Minimal bilingual (English / Chinese) string table.

Wrap UI strings with :func:`tr` and call :func:`set_language` to switch the
whole interface at runtime.
"""
LANG = "en"

STRINGS = {
    "en": {
        "file": "&File",
        "open": "&Open...",
        "save_as": "&Save As...",
        "save_pdf": "Save as &PDF...",
        "scan": "Scan &Document...",
        "exit": "E&xit",
        "edit": "&Edit",
        "undo": "&Undo",
        "redo": "&Redo",
        "view": "&View",
        "fit": "&Fit to Window",
        "actual": "&Actual Size (100%)",
        "language": "&Language",
        "lang_en": "English",
        "lang_zh": "中文",
        "compare": "&Compare",
        "cmp_original": "&Original",
        "cmp_result": "&Result",
        "cmp_split": "&Split View",
        "cmp_drag": "&Drag Line",
        "theme": "&Theme",
        "theme_light": "&Light",
        "theme_dark": "&Dark",
        "adjust": "Adjust",
        "operation": "Operation:",
        "apply": "Apply",
        "reset": "Reset",
        "no_image": "No image",
        "remove_bg": "Remove &Background...",
        "bg_method": "Method:",
        "grabcut": "GrabCut (draw rectangle)",
        "threshold": "Color threshold (click background)",
        "bg_output": "Background:",
        "bg_white": "White",
        "bg_transparent": "Transparent",
        "grabcut_hint": "Draw a rectangle around the subject",
        "threshold_hint": "Click a background point",
        "select_hint": "Click regions to keep / remove",
        "keep_all": "Keep All",
        "keep_largest": "Keep Largest",
        "remove_all": "Remove All",
        "cancel": "Cancel",
        "run_again": "Run Again",
        "please_segment": "Please segment the image first.",
    },
    "zh": {
        "file": "文件(&F)",
        "open": "打开(&O)...",
        "save_as": "另存为(&A)...",
        "save_pdf": "另存为 PDF(&P)...",
        "scan": "扫描文档(&D)...",
        "exit": "退出(&X)",
        "edit": "编辑(&E)",
        "undo": "撤销(&U)",
        "redo": "重做(&R)",
        "view": "视图(&V)",
        "fit": "适应窗口(&F)",
        "actual": "实际大小 100%(&A)",
        "language": "语言(&L)",
        "lang_en": "English",
        "lang_zh": "中文",
        "compare": "对比(&C)",
        "cmp_original": "原图(&O)",
        "cmp_result": "效果图(&R)",
        "cmp_split": "左右分屏(&S)",
        "cmp_drag": "拖动对比线(&D)",
        "theme": "主题(&T)",
        "theme_light": "浅色(&L)",
        "theme_dark": "深色(&D)",
        "adjust": "调整",
        "operation": "操作:",
        "apply": "应用",
        "reset": "重置",
        "no_image": "无图片",
        "remove_bg": "去除背景(&B)...",
        "bg_method": "方法:",
        "grabcut": "GrabCut（框选主体）",
        "threshold": "颜色阈值（点选背景）",
        "bg_output": "背景:",
        "bg_white": "白色",
        "bg_transparent": "透明",
        "grabcut_hint": "请框选主体",
        "threshold_hint": "请点选背景点",
        "select_hint": "点击区域切换保留/删除",
        "keep_all": "全保留",
        "keep_largest": "只保留最大块",
        "remove_all": "全部删除",
        "cancel": "取消",
        "run_again": "重新分割",
        "please_segment": "请先进行分割。",
    },
}


def tr(key: str) -> str:
    return STRINGS.get(LANG, {}).get(key, key)


def set_language(lang: str) -> None:
    global LANG
    if lang in ("en", "zh"):
        LANG = lang


def language() -> str:
    return LANG
