"""
Test de la interfaz Streamlit con Playwright.
Verifica que cada combinación metodo × modelo pueda lanzar una simulación
y que la app muestre el mensaje de éxito.

Correr:
    conda run -n bfr_env python -m pytest tests/test_ui_playwright.py -v
"""
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8502"
TIMEOUT = 120_000


def wait_streamlit_ready(page):
    """Espera a que Streamlit termine de renderizar (título visible)."""
    page.wait_for_selector("text=ADR / MRMT Transport Simulator", timeout=15_000)
    # Espera adicional para que los widgets del sidebar se estabilicen
    page.wait_for_timeout(500)


def open_sidebar(page):
    """Abre el sidebar (botón >> en la esquina superior izquierda)."""
    # Verifica que el sidebar está cerrado (selects fuera del viewport)
    sel = page.locator("[data-baseweb='select']").first
    bb = sel.bounding_box()
    if bb and bb["x"] < 0:
        page.mouse.click(32, 30)
        page.wait_for_timeout(800)


def select_streamlit(page, aria_label_fragment, value):
    """Clica el contenedor data-baseweb='select' que contiene el input con
    el aria-label dado, luego selecciona la opción `value`."""
    container = page.locator(
        f"xpath=//input[contains(@aria-label,'{aria_label_fragment}')]"
        f"/ancestor::div[@data-baseweb='select']"
    )
    container.wait_for(state="visible", timeout=10_000)
    container.click()
    page.get_by_role("option", name=value, exact=True).click()


@pytest.fixture(scope="session")
def browser_ctx():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 900})
        yield ctx
        ctx.close()
        browser.close()


def run_simulation(page, metodo, model):
    page.goto(BASE_URL, timeout=30_000)
    wait_streamlit_ready(page)
    open_sidebar(page)

    select_streamlit(page, "Numerical Method", metodo)
    select_streamlit(page, "Transport Model", model)
    select_streamlit(page, "Run Type", "standard")

    # Tab Run → verificar config válida → lanzar
    page.get_by_role("tab", name="🚀 Run").click()
    page.wait_for_selector("text=Configuration valid", timeout=8_000)
    page.get_by_role("button", name="🚀 Run Simulation").click()

    # Esperar resultado
    ok_loc  = page.locator("text=Simulación terminada correctamente")
    err_loc = page.locator("text=Se detectaron errores")
    ok_loc.or_(err_loc).wait_for(timeout=TIMEOUT)
    return ok_loc.count() > 0


CASES = [
    ("bfr",     "adr"),
    ("bfr",     "mrmt_semi"),
    ("bfr",     "mrmt_block"),
    ("fenicsx", "adr"),
    ("fenicsx", "mrmt_semi"),
    ("fenicsx", "mrmt_block"),
]


@pytest.mark.parametrize("metodo,model", CASES, ids=[f"{m}-{mo}" for m, mo in CASES])
def test_run(browser_ctx, metodo, model):
    page = browser_ctx.new_page()
    try:
        ok = run_simulation(page, metodo, model)
        assert ok, f"La simulación {metodo}/{model} reportó error en la UI"
    finally:
        page.close()
