from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


DOCX_PATH = Path("/Users/ruslanmuradov/vkr/вкр.docx")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = f"{{{NS['w']}}}"

SECTION_HEADING = "Разработка пользовательского приложения для регистрации походки и визуализации результатов исследований"
REFERENCES_HEADING = "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ"

SECTION_BLOCK = [
    ("h1", SECTION_HEADING),
    (
        "p",
        "Разрабатываемое пользовательское приложение предназначено для регистрации походки по данным RGBD-камеры, автоматического расчёта базовых параметров движения и визуализации результатов исследования в удобной для оператора форме. В отличие от лабораторных систем, ориентированных прежде всего на получение высокоточного эталонного результата, данное приложение рассматривается как прикладной интерфейс биотехнической системы, обеспечивающий подготовку сеанса, запись данных, контроль качества, последующую обработку и просмотр архива измерений.",
    ),
    ("h2", "8.1 Требования к приложению"),
    (
        "p",
        "К приложению предъявляются требования, связанные как с процессом регистрации данных, так и с обеспечением воспроизводимости исследований. Приложение должно поддерживать подключение Intel RealSense D455f, синхронную запись цветового и глубинного потоков, ввод метаданных испытуемого и протокола, автоматическое сохранение результатов сеанса и последующий доступ к архиву исследований. Кроме того, требуется наглядный интерфейс, в котором оператор может видеть предпросмотр RGB и depth, текущий статус захвата и итоговые численные показатели.",
    ),
    (
        "p",
        "С точки зрения алгоритмической части важна поддержка автоматического расчёта 3D-координат ключевых точек тела и производных метрик походки. Для исследовательского применения необходимо сохранять не только итоговые показатели, но и промежуточные данные, позволяющие повторно выполнить обработку или верифицировать корректность вычислений. Поэтому структура хранения должна включать метаданные, покадровые координаты ключевых точек, агрегированные показатели сеанса и по возможности сырой архив RGBD-потока.",
    ),
    ("h2", "8.2 Архитектура программного решения"),
    (
        "p",
        "Приложение реализовано по модульному принципу и включает модуль захвата данных, модуль оценки позы, модуль аналитической обработки и модуль визуализации. Модуль захвата использует официальный SDK Intel RealSense и Python-обёртку pyrealsense2 для получения синхронных RGB и depth кадров, а также для записи полного сеанса в формате .bag [40]. Такой подход позволяет разделить задачу регистрации данных и задачу их последующей аналитической обработки.",
    ),
    (
        "p",
        "Модуль оценки позы реализует гибридную схему RGBD-обработки. Сначала по RGB-изображению определяется 2D-положение ключевых точек тела с помощью MediaPipe Pose [39]. Затем для каждой ключевой точки оценивается глубина по локальной области карты глубины, и по внутренним параметрам камеры вычисляются метрические 3D-координаты. Благодаря этому из одной RGBD-камеры формируется компактное представление походки в виде временного ряда пространственных координат суставов.",
    ),
    (
        "p",
        "Модуль аналитики рассчитывает траекторию таза, относительное положение стоп, события шага, скорость, каденс, среднюю длину шага и угловые зависимости для основных суставов нижних конечностей. Пользовательский интерфейс реализован как локальное desktop-приложение на tkinter с использованием matplotlib для построения графиков. Такое решение достаточно для исследовательского стенда, не требует серверной инфраструктуры и хорошо соответствует задаче локальной работы с медицинскими данными.",
    ),
    ("h2", "8.3 Основные сценарии работы"),
    (
        "p",
        "Работа с приложением начинается с создания нового сеанса, ввода сведений об испытуемом и выбора протокола. После этого оператор запускает предпросмотр, контролирует положение камеры, рабочую дистанцию и попадание испытуемого в кадр. На этапе записи приложение регистрирует RGBD-поток, а при доступности модуля оценки позы параллельно формирует 3D-координаты ключевых точек и рассчитывает предварительные показатели походки.",
    ),
    (
        "p",
        "После завершения сеанса приложение сохраняет метаданные, покадровые координаты и итоговые метрики, затем предоставляет средства для просмотра архива. На вкладке результатов пользователь может выбрать любой сохранённый проход, получить сводку параметров, просмотреть траекторию движения таза, графики сепарации стоп и угловые кривые для коленного и тазобедренного суставов. Дополнительно в приложении предусмотрен демонстрационный режим, позволяющий проверять интерфейс и формат данных даже без подключённой камеры.",
    ),
    ("h2", "8.4 Форматы данных и визуализация результатов"),
    (
        "p",
        "Для обеспечения воспроизводимости исследования данные сохраняются в нескольких взаимосвязанных файлах. Файл metadata.json содержит сведения о сеансе и условиях регистрации, файл landmarks.csv хранит покадровые 3D-координаты ключевых точек, их видимость и положение в плоскости изображения, а файл metrics.json содержит итоговые агрегированные показатели. При работе с физической камерой дополнительно сохраняется файл recording.bag, содержащий сырой синхронный RGBD-поток.",
    ),
    (
        "p",
        "Визуализация строится вокруг двух задач: контроля качества регистрации и интерпретации результата. Во время записи отображаются RGB-кадр, карта глубины и ключевые точки тела. После завершения сеанса формируются графики траектории таза, временной зависимости относительного положения стоп и изменения суставных углов. Наличие одновременно численных показателей и графической интерпретации повышает удобство использования системы в исследовательской и учебной практике.",
    ),
    ("h2", "8.5 Выбор Intel RealSense D455f для экспериментальной установки"),
    (
        "p",
        "В качестве базового сенсора выбрана камера Intel RealSense D455f. По официальной спецификации Intel устройство относится к стереоскопическим RGBD-камерам, поддерживает рабочий диапазон 0,6–6 м, глубинный поток до 1280×720 при частоте до 90 кадров/с, поле зрения глубинного канала 86° × 57° и интерфейс USB 3.1 [35]. Эти характеристики соответствуют задаче регистрации прямолинейной ходьбы в помещении, когда требуется наблюдать полный проход испытуемого в пределах нескольких метров.",
    ),
    (
        "p",
        "При сравнении с альтернативами выбранная камера представляет практический компромисс между вычислительной доступностью и достаточностью метрических характеристик. Orbbec Femto Bolt позиционируется как замена Azure Kinect DK и использует ToF-подход, обеспечивая глубинный диапазон 0,25–5,46 м и 4K RGB-канал [36]. Azure Kinect DK также предлагает ToF-сенсор глубины, 12-Мп RGB-камеру и IMU [37]. Stereolabs ZED 2i поддерживает большие дистанции и высокую дальность стереозрения, но ориентирована на более ресурсоёмкие конфигурации [38]. Для данной ВКР, где требуется разработать прикладное локальное приложение и выполнить первичную регистрацию походки, Intel RealSense D455f является рациональным выбором благодаря зрелому SDK, высокой частоте depth-потока и доступности оборудования.",
    ),
]

REFERENCE_BLOCK = [
    "Intel. Intel® RealSense™ Depth Camera D455f Specifications. – URL: https://www.intel.com/content/www/us/en/products/sku/233193/intel-realsense-depth-camera-d455f/specifications.html",
    "Orbbec. Femto Bolt. – URL: https://store.orbbec.com/products/femto-bolt",
    "Microsoft. Azure Kinect DK. – URL: https://azure.microsoft.com/en-us/products/kinect-dk/",
    "Stereolabs. ZED 2i Stereo Camera. – URL: https://www.stereolabs.com/store/products/zed-2i",
    "Google AI Edge. MediaPipe Pose. – URL: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md",
    "Intel RealSense. librealsense. – URL: https://github.com/IntelRealSense/librealsense",
]


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()


def find_paragraph(body: ET.Element, text: str) -> ET.Element:
    for paragraph in body.findall("w:p", NS):
        if paragraph_text(paragraph) == text:
            return paragraph
    raise ValueError(f"Paragraph not found: {text}")


def build_paragraph(template: ET.Element, text: str) -> ET.Element:
    paragraph = copy.deepcopy(template)
    for child in list(paragraph):
        if child.tag != f"{W}pPr":
            paragraph.remove(child)
    run = ET.SubElement(paragraph, f"{W}r")
    text_node = ET.SubElement(run, f"{W}t")
    if text.startswith(" ") or text.endswith(" "):
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text
    return paragraph


def replace_paragraph_text(paragraph: ET.Element, text: str) -> None:
    for child in list(paragraph):
        if child.tag != f"{W}pPr":
            paragraph.remove(child)
    run = ET.SubElement(paragraph, f"{W}r")
    text_node = ET.SubElement(run, f"{W}t")
    text_node.text = text


def insert_block(body: ET.Element) -> bool:
    if any(paragraph_text(paragraph) == SECTION_HEADING for paragraph in body.findall("w:p", NS)):
        return False

    bibliography = find_paragraph(body, REFERENCES_HEADING)
    heading1_template = find_paragraph(body, "Разработка алгоритмов оценки 3D‑позы по видеоизображению")
    heading2_template = find_paragraph(body, "7.1 Постановка задачи и исходные данные")
    normal_template = find_paragraph(body, "Для интеграции с пользовательским приложением видеорегистрации результат алгоритма целесообразно стандартизировать в виде структуры кадра 3D-позы, содержащей временную метку, список суставов с координатами и уверенностями, а также интегральную оценку качества  и диагностические флаги. Дополнительно могут формироваться прикладные события (например, шаг, контакт стопы, повтор упражнения) и сохраняться параметры обработки (выбранная модель 2D-позы, параметры фильтрации, версия калибровки), чтобы обеспечить воспроизводимость исследования. Для хранения и последующего анализа результаты удобно сохранять в (JSON/CSV) для отчётов и в базе данных (например, SQLite) для архива сессий, а визуализацию реализовать через наложение 2D-скелета на RGB и отображение 3D-скелета в отдельном интерактивном окне. Такой формат хорошо соответствует требованиям цифровых реабилитационных систем, где важны и наглядность, и трассируемость параметров обработки.")

    children = list(body)
    insert_index = children.index(bibliography)
    for style, text in SECTION_BLOCK:
        template = normal_template
        if style == "h1":
            template = heading1_template
        elif style == "h2":
            template = heading2_template
        body.insert(insert_index, build_paragraph(template, text))
        insert_index += 1
    return True


def append_references(body: ET.Element) -> int:
    normalized = 0
    for paragraph in body.findall("w:p", NS):
        text = paragraph_text(paragraph)
        for index, reference in enumerate(REFERENCE_BLOCK, start=35):
            legacy = f"{index}. {reference}"
            if text == legacy:
                replace_paragraph_text(paragraph, reference)
                normalized += 1

    existing = {paragraph_text(paragraph) for paragraph in body.findall("w:p", NS)}
    normal_template = find_paragraph(body, "Всемирная организация здравоохранения. Реабилитация.  – URL: https://www.who.int/news-room/fact-sheets/detail/rehabilitation")
    section_properties = body.find("w:sectPr", NS)
    insert_index = len(list(body)) if section_properties is None else list(body).index(section_properties)
    added = 0
    for reference in REFERENCE_BLOCK:
        if reference in existing:
            continue
        body.insert(insert_index, build_paragraph(normal_template, reference))
        insert_index += 1
        added += 1
    return added + normalized


def main() -> None:
    with ZipFile(DOCX_PATH, "r") as archive:
        document_xml = archive.read("word/document.xml")
        file_map = {name: archive.read(name) for name in archive.namelist() if name != "word/document.xml"}

    root = ET.fromstring(document_xml)
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("word/document.xml has no body")

    inserted = insert_block(body)
    added_refs = append_references(body)
    if not inserted and added_refs == 0:
        print("No changes needed")
        return

    updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    with ZipFile(temp_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in file_map.items():
            archive.writestr(name, data)
        archive.writestr("word/document.xml", updated_xml)

    temp_path.replace(DOCX_PATH)
    print(f"Updated {DOCX_PATH.name}: inserted_section={inserted}, added_refs={added_refs}")


if __name__ == "__main__":
    main()
