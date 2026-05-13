import json
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Docling / RAG Extraction Viewer",
    layout="wide",
)

UPLOADS_ROOT = Path("/app/uploads")
EXTRACTED_ROOT = Path("/app/extracted_files")


# -----------------------------
# File discovery
# -----------------------------

def find_pdfs() -> list[Path]:
    return sorted(UPLOADS_ROOT.rglob("*.pdf"))


def find_jsons(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(EXTRACTED_ROOT.rglob(pattern))
    return sorted(set(files))


def find_document_jsons() -> list[Path]:
    return find_jsons([
        "document.json",
        "docling_document.json",
        "*document*.json",
    ])


def find_normalized_jsons() -> list[Path]:
    return find_jsons([
        "normalized.json",
        "normalized_units.json",
        "*normalized*.json",
    ])


def find_rag_jsons() -> list[Path]:
    return find_jsons([
        "rag_units.json",
        "final_rag_output.json",
        "*rag*.json",
    ])


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_file_id_from_pdf(pdf_path: Path) -> str | None:
    """
    uploads/{file_id}/{uuid}.pdf
    """
    try:
        relative = pdf_path.relative_to(UPLOADS_ROOT)
        return relative.parts[0]
    except Exception:
        return None


def get_chat_file_from_extracted(json_path: Path) -> tuple[str | None, str | None]:
    """
    extracted_files/{chat_id}/{file_id}/{json}
    """
    try:
        relative = json_path.relative_to(EXTRACTED_ROOT)
        chat_id = relative.parts[0] if len(relative.parts) > 0 else None
        file_id = relative.parts[1] if len(relative.parts) > 1 else None
        return chat_id, file_id
    except Exception:
        return None, None


def find_matching_json_for_pdf(pdf_path: Path, json_files: list[Path]) -> Path | None:
    file_id = get_file_id_from_pdf(pdf_path)
    if not file_id:
        return None

    for json_file in json_files:
        _, extracted_file_id = get_chat_file_from_extracted(json_file)
        if extracted_file_id == file_id:
            return json_file

    return None


# -----------------------------
# JSON normalization
# -----------------------------

def normalize_units(raw: Any) -> list[dict[str, Any]]:
    """
    Accepts:
    - list[dict]
    - {"rag_units": [...]}
    - {"normalized": [...]}
    - Docling-style document JSON with texts/tables/pictures/groups
    """
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if not isinstance(raw, dict):
        return []

    for key in ["rag_units", "normalized", "items", "elements", "chunks"]:
        value = raw.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    # Basic Docling document JSON fallback
    units: list[dict[str, Any]] = []

    for key, item_type in [
        ("texts", "text"),
        ("tables", "table"),
        ("pictures", "picture"),
        ("groups", "group"),
    ]:
        values = raw.get(key)
        if isinstance(values, list):
            for idx, item in enumerate(values):
                if not isinstance(item, dict):
                    continue

                self_ref = item.get("self_ref") or item.get("$ref") or f"#/{key}/{idx}"

                unit = {
                    "id": self_ref,
                    "source_ref": self_ref,
                    "self_ref": self_ref,
                    "order": item.get("order", idx),
                    "type": item.get("type") or item_type,
                    "label": item.get("label"),
                    "text": item.get("text") or item.get("orig") or "",
                    "page_no": extract_page_no_from_docling_item(item),
                    "bbox": extract_bbox_from_docling_item(item),
                    "raw": item,
                }

                units.append(unit)

    return sorted(units, key=lambda x: x.get("order", 0))


def extract_page_no_from_docling_item(item: dict[str, Any]) -> int | None:
    prov = item.get("prov")
    if isinstance(prov, list) and prov:
        first = prov[0]
        if isinstance(first, dict):
            return first.get("page_no")

    return item.get("page_no")


def extract_bbox_from_docling_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(item.get("bbox"), dict):
        return item["bbox"]

    prov = item.get("prov")
    if isinstance(prov, list) and prov:
        first = prov[0]
        if isinstance(first, dict) and isinstance(first.get("bbox"), dict):
            return first["bbox"]

    return None


def source_key(unit: dict[str, Any]) -> str:
    return str(
        unit.get("source_ref")
        or unit.get("self_ref")
        or unit.get("id")
        or unit.get("ref")
        or ""
    )


def text_len(unit: dict[str, Any]) -> int:
    return len(str(unit.get("text") or ""))


def make_summary_df(units: list[dict[str, Any]], label: str) -> pd.DataFrame:
    rows = []

    for u in units:
        rows.append(
            {
                "json": label,
                "order": u.get("order"),
                "type": u.get("type"),
                "label": u.get("label"),
                "page_no": u.get("page_no"),
                "source_ref": source_key(u),
                "heading": u.get("heading"),
                "text_len": text_len(u),
                "has_bbox": bool(u.get("bbox")),
                "has_image": bool(u.get("image_path")),
                "has_table_markdown": bool(u.get("table_markdown")),
                "has_table_vision": bool(u.get("table_vision")),
                "has_vision_metadata": bool(u.get("vision_metadata")),
                "text_preview": str(u.get("text") or "")[:180],
            }
        )

    return pd.DataFrame(rows)


def compare_units(
        document_units: list[dict[str, Any]],
        normalized_units: list[dict[str, Any]],
        rag_units: list[dict[str, Any]],
) -> pd.DataFrame:
    doc_map = {source_key(u): u for u in document_units if source_key(u)}
    norm_map = {source_key(u): u for u in normalized_units if source_key(u)}
    rag_map = {source_key(u): u for u in rag_units if source_key(u)}

    all_keys = sorted(set(doc_map) | set(norm_map) | set(rag_map))

    rows = []

    for key in all_keys:
        d = doc_map.get(key)
        n = norm_map.get(key)
        r = rag_map.get(key)

        rows.append(
            {
                "source_ref": key,
                "in_document": d is not None,
                "in_normalized": n is not None,
                "in_rag_units": r is not None,

                "document_type": d.get("type") if d else None,
                "normalized_type": n.get("type") if n else None,
                "rag_type": r.get("type") if r else None,

                "document_page": d.get("page_no") if d else None,
                "normalized_page": n.get("page_no") if n else None,
                "rag_page": r.get("page_no") if r else None,

                "document_text_len": text_len(d) if d else 0,
                "normalized_text_len": text_len(n) if n else 0,
                "rag_text_len": text_len(r) if r else 0,

                "normalized_has_bbox": bool(n and n.get("bbox")),
                "rag_has_bbox": bool(r and r.get("bbox")),

                "normalized_has_image": bool(n and n.get("image_path")),
                "rag_has_image": bool(r and r.get("image_path")),

                "normalized_table_md": bool(n and n.get("table_markdown")),
                "rag_table_md": bool(r and r.get("table_markdown")),

                "rag_has_table_vision": bool(r and r.get("table_vision")),
                "rag_has_vision_metadata": bool(r and r.get("vision_metadata")),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# PDF rendering
# -----------------------------

def render_pdf_page(pdf_path: Path, page_no: int, scale: int = 2) -> tuple[Image.Image, float]:
    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image, page.rect.height


def draw_bbox(
        image: Image.Image,
        bbox: dict,
        page_height: float,
        scale: int = 2,
        outline: str = "red",
) -> Image.Image:
    if not bbox:
        return image

    draw = ImageDraw.Draw(image)

    l = bbox.get("l")
    t = bbox.get("t")
    r = bbox.get("r")
    b = bbox.get("b")

    if None in [l, t, r, b]:
        return image

    coord_origin = bbox.get("coord_origin")

    x1 = l * scale
    x2 = r * scale

    if coord_origin == "BOTTOMLEFT":
        y1 = (page_height - t) * scale
        y2 = (page_height - b) * scale
    else:
        y1 = t * scale
        y2 = b * scale

    draw.rectangle([x1, y1, x2, y2], outline=outline, width=3)

    return image


def resolve_image_path(image_path: str | None) -> Path | None:
    if not image_path:
        return None

    path = Path(image_path)

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        Path("/app") / image_path,
        EXTRACTED_ROOT / image_path,
        UPLOADS_ROOT / image_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


# -----------------------------
# UI
# -----------------------------

st.title("Docling JSON Compare Viewer")

pdf_files = find_pdfs()
document_jsons = find_document_jsons()
normalized_jsons = find_normalized_jsons()
rag_jsons = find_rag_jsons()

if not pdf_files:
    st.error("No PDF files found under /app/uploads.")
    st.stop()

if not document_jsons and not normalized_jsons and not rag_jsons:
    st.error("No JSON files found under /app/extracted_files.")
    st.stop()

st.sidebar.header("Select PDF")

pdf_path = st.sidebar.selectbox(
    "PDF",
    pdf_files,
    format_func=lambda p: str(p.relative_to(UPLOADS_ROOT)),
)

matched_document_json = find_matching_json_for_pdf(pdf_path, document_jsons)
matched_normalized_json = find_matching_json_for_pdf(pdf_path, normalized_jsons)
matched_rag_json = find_matching_json_for_pdf(pdf_path, rag_jsons)


def select_json(label: str, files: list[Path], matched: Path | None, root: Path) -> Path | None:
    if not files:
        st.sidebar.warning(f"No {label} JSON found.")
        return None

    index = files.index(matched) if matched in files else 0

    return st.sidebar.selectbox(
        label,
        files,
        index=index,
        format_func=lambda p: str(p.relative_to(root)),
    )


st.sidebar.header("Select JSONs")

document_json_path = select_json(
    "Document JSON",
    document_jsons,
    matched_document_json,
    EXTRACTED_ROOT,
)

normalized_json_path = select_json(
    "Normalized JSON",
    normalized_jsons,
    matched_normalized_json,
    EXTRACTED_ROOT,
)

rag_json_path = select_json(
    "RAG Units JSON",
    rag_jsons,
    matched_rag_json,
    EXTRACTED_ROOT,
)

pdf_file_id = get_file_id_from_pdf(pdf_path)
st.sidebar.markdown("### File ID")
st.sidebar.write("PDF file_id:", pdf_file_id)

for label, path in [
    ("document", document_json_path),
    ("normalized", normalized_json_path),
    ("rag", rag_json_path),
]:
    if path:
        chat_id, extracted_file_id = get_chat_file_from_extracted(path)
        st.sidebar.write(f"{label} chat_id:", chat_id)
        st.sidebar.write(f"{label} file_id:", extracted_file_id)
        if pdf_file_id and extracted_file_id and pdf_file_id != extracted_file_id:
            st.sidebar.warning(f"{label} JSON file_id does not match PDF file_id.")

document_raw = load_json(document_json_path) if document_json_path else None
normalized_raw = load_json(normalized_json_path) if normalized_json_path else None
rag_raw = load_json(rag_json_path) if rag_json_path else None

document_units = normalize_units(document_raw) if document_raw is not None else []
normalized_units = normalize_units(normalized_raw) if normalized_raw is not None else []
rag_units = normalize_units(rag_raw) if rag_raw is not None else []

tabs = st.tabs([
    "Visual Page",
    "Document JSON",
    "Normalized JSON",
    "RAG Units JSON",
    "Compare",
    "Raw JSON",
])

# -----------------------------
# Tab 1: visual page
# -----------------------------

with tabs[0]:
    st.subheader("Visual page with bounding boxes")

    active_source = st.radio(
        "Show boxes from",
        ["rag_units", "normalized", "document"],
        horizontal=True,
    )

    if active_source == "rag_units":
        active_units = rag_units
    elif active_source == "normalized":
        active_units = normalized_units
    else:
        active_units = document_units

    page_numbers = sorted(
        {
            u.get("page_no")
            for u in active_units
            if isinstance(u.get("page_no"), int)
        }
    )

    if not page_numbers:
        st.warning(f"No page_no found for {active_source}.")
    else:
        col_a, col_b, col_c = st.columns([1, 1, 2])

        with col_a:
            page_no = st.selectbox("Page", page_numbers)

        with col_b:
            types = ["all"] + sorted({str(u.get("type")) for u in active_units if u.get("type")})
            selected_type = st.selectbox("Type", types)

        with col_c:
            query = st.text_input("Text contains", "")

        page_units = [u for u in active_units if u.get("page_no") == page_no]

        if selected_type != "all":
            page_units = [u for u in page_units if u.get("type") == selected_type]

        if query.strip():
            q = query.lower().strip()
            page_units = [
                u for u in page_units
                if q in str(u.get("text", "")).lower()
                   or q in str(u.get("heading", "")).lower()
                   or q in str(u.get("caption", "")).lower()
                   or q in str(u.get("table_markdown", "")).lower()
            ]

        left, right = st.columns([1.2, 1])

        with left:
            page_img, page_height = render_pdf_page(pdf_path, page_no, scale=2)

            for u in page_units:
                bbox = u.get("bbox")
                if bbox:
                    page_img = draw_bbox(page_img, bbox, page_height, scale=2)

            st.image(page_img, use_container_width=True)

        with right:
            st.write(f"Showing {len(page_units)} elements from `{active_source}`")

            if page_units:
                st.dataframe(
                    make_summary_df(page_units, active_source),
                    use_container_width=True,
                )

            for idx, u in enumerate(page_units):
                src = source_key(u)
                with st.expander(f"{idx + 1}. {u.get('type')} | {src}"):
                    st.write("Order:", u.get("order"))
                    st.write("Label:", u.get("label"))
                    st.write("Heading:", u.get("heading"))
                    st.write("Heading path:", u.get("heading_path"))
                    st.write("Page:", u.get("page_no"))
                    st.write("BBox:", u.get("bbox"))
                    st.write("Image path:", u.get("image_path"))

                    st.markdown("#### Text")
                    st.text((u.get("text") or "")[:8000])

                    if u.get("table_markdown"):
                        st.markdown("#### Table markdown")
                        st.code(u.get("table_markdown"), language="markdown")

                    if u.get("key_findings"):
                        st.markdown("#### Key findings")
                        for finding in u.get("key_findings") or []:
                            st.write("- " + str(finding))

                    if u.get("rag_keywords"):
                        st.markdown("#### RAG keywords")
                        st.write(", ".join(map(str, u.get("rag_keywords") or [])))

                    if u.get("caption"):
                        st.markdown("#### Caption")
                        st.write(u.get("caption"))

                    if u.get("table_vision"):
                        st.markdown("#### Table vision")
                        st.json(u.get("table_vision"))

                    if u.get("vision_metadata"):
                        st.markdown("#### Vision metadata")
                        st.json(u.get("vision_metadata"))

                    resolved_image_path = resolve_image_path(u.get("image_path"))
                    if resolved_image_path:
                        st.markdown("#### Extracted image")
                        st.image(str(resolved_image_path), use_container_width=True)

# -----------------------------
# Tab 2: document summary
# -----------------------------

with tabs[1]:
    st.subheader("Document JSON")
    st.write("Path:", str(document_json_path) if document_json_path else "None")
    st.write("Units:", len(document_units))

    if document_units:
        st.dataframe(
            make_summary_df(document_units, "document"),
            use_container_width=True,
        )

# -----------------------------
# Tab 3: normalized summary
# -----------------------------

with tabs[2]:
    st.subheader("Normalized JSON")
    st.write("Path:", str(normalized_json_path) if normalized_json_path else "None")
    st.write("Units:", len(normalized_units))

    if normalized_units:
        st.dataframe(
            make_summary_df(normalized_units, "normalized"),
            use_container_width=True,
        )

# -----------------------------
# Tab 4: rag summary
# -----------------------------

with tabs[3]:
    st.subheader("RAG Units JSON")
    st.write("Path:", str(rag_json_path) if rag_json_path else "None")
    st.write("Units:", len(rag_units))

    if rag_units:
        st.dataframe(
            make_summary_df(rag_units, "rag_units"),
            use_container_width=True,
        )

        st.markdown("### RAG type counts")
        df = make_summary_df(rag_units, "rag_units")
        st.dataframe(
            df.groupby("type").size().reset_index(name="count"),
            use_container_width=True,
        )

# -----------------------------
# Tab 5: compare
# -----------------------------

with tabs[4]:
    st.subheader("Compare document → normalized → rag_units")

    compare_df = compare_units(
        document_units=document_units,
        normalized_units=normalized_units,
        rag_units=rag_units,
    )

    st.write("Compared refs:", len(compare_df))

    if not compare_df.empty:
        summary = {
            "document_units": len(document_units),
            "normalized_units": len(normalized_units),
            "rag_units": len(rag_units),
            "missing_in_normalized": int((~compare_df["in_normalized"]).sum()),
            "missing_in_rag_units": int((~compare_df["in_rag_units"]).sum()),
            "rag_tables_with_markdown": int(compare_df["rag_table_md"].sum()),
            "rag_with_table_vision": int(compare_df["rag_has_table_vision"].sum()),
            "rag_with_vision_metadata": int(compare_df["rag_has_vision_metadata"].sum()),
        }

        st.json(summary)

        filter_mode = st.selectbox(
            "Compare filter",
            [
                "all",
                "missing in normalized",
                "missing in rag_units",
                "tables",
                "rag has table markdown",
                "rag missing table markdown",
                "rag has vision metadata",
            ],
        )

        filtered_df = compare_df

        if filter_mode == "missing in normalized":
            filtered_df = compare_df[~compare_df["in_normalized"]]
        elif filter_mode == "missing in rag_units":
            filtered_df = compare_df[~compare_df["in_rag_units"]]
        elif filter_mode == "tables":
            filtered_df = compare_df[
                (compare_df["document_type"] == "table")
                | (compare_df["normalized_type"] == "table")
                | (compare_df["rag_type"] == "table")
                ]
        elif filter_mode == "rag has table markdown":
            filtered_df = compare_df[compare_df["rag_table_md"]]
        elif filter_mode == "rag missing table markdown":
            filtered_df = compare_df[
                (compare_df["rag_type"] == "table") & (~compare_df["rag_table_md"])
                ]
        elif filter_mode == "rag has vision metadata":
            filtered_df = compare_df[compare_df["rag_has_vision_metadata"]]

        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("### Inspect one source_ref")

        source_refs = filtered_df["source_ref"].dropna().tolist()
        if source_refs:
            selected_ref = st.selectbox("source_ref", source_refs)

            doc_map = {source_key(u): u for u in document_units if source_key(u)}
            norm_map = {source_key(u): u for u in normalized_units if source_key(u)}
            rag_map = {source_key(u): u for u in rag_units if source_key(u)}

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("#### Document")
                if selected_ref in doc_map:
                    st.json(doc_map[selected_ref])
                else:
                    st.warning("Not found in document JSON.")

            with c2:
                st.markdown("#### Normalized")
                if selected_ref in norm_map:
                    st.json(norm_map[selected_ref])
                else:
                    st.warning("Not found in normalized JSON.")

            with c3:
                st.markdown("#### RAG Unit")
                if selected_ref in rag_map:
                    st.json(rag_map[selected_ref])
                else:
                    st.warning("Not found in rag_units JSON.")

# -----------------------------
# Tab 6: raw json
# -----------------------------

with tabs[5]:
    st.subheader("Raw JSON")

    raw_choice = st.selectbox(
        "Raw JSON to view",
        ["document", "normalized", "rag_units"],
    )

    if raw_choice == "document":
        st.json(document_raw)
    elif raw_choice == "normalized":
        st.json(normalized_raw)
    else:
        st.json(rag_raw)
