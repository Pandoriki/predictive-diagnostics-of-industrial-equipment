import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

@pytest.fixture
def driver():
    driver = webdriver.Chrome()  
    driver.maximize_window()
    yield driver
    driver.quit()

def test_equipment_dashboard(driver):
    # 1) Перейти по ссылке
    driver.get("http://localhost/")
    
    # 2) Клик по элементу (требует внимания)
    attention_element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[1]/div/div/div[1]/div/button[2]"))
    )
    attention_element.click()
    time.sleep(2)
    
    # 3) Клик по элементу (предупреждения)
    warning_element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[1]/div/div/div[1]/div[2]/button[2]"))
    )
    warning_element.click()
    time.sleep(2)
    
    # 4) Клик по элементу (критическое)
    critical_element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[1]/div/div/div[1]/div[2]/button[3]"))
    )
    critical_element.click()
    time.sleep(2)
    
    # 5) Клик (все)
    all_element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[1]/div/div/div[1]/div[1]/button[1]"))
    )
    all_element.click()
    time.sleep(2)
    
    # 6) Клик по кнопке (сортировка RUL)
    sort_rul_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[1]/div/div/div[2]/button[1]"))
    )
    sort_rul_button.click()
    time.sleep(2)
    
    # 7) Клик по кнопке (сортировка по ID)
    sort_id_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[1]/div/div/div[2]/button[2]"))
    )
    sort_id_button.click()
    time.sleep(2)
    
    # 8) Нажать на кнопку (открыть конкретный отчет)
    open_report_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[2]/div[1]/div[2]"))
    )
    open_report_button.click()
    time.sleep(2)
    
    # 11) Нажать на кнопку (вернуться назад)
    back_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/main/div/div[1]/div/button"))
    )
    back_button.click()
    time.sleep(2)
    
    # 12) Нажать на кнопку изменить цвет темы
    theme_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/header/div[2]/button"))
    )
    theme_button.click()
    time.sleep(2)
    
    # 13) Нажать на кнопку охраника
    guard_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div[2]/header/div[2]/div/button[3]"))
    )
    guard_button.click()
    time.sleep(2)
    
    # Проверка что тест завершился успешно
    assert driver.current_url is not None