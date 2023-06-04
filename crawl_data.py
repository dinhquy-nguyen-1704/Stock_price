import pandas as pd
import os
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
import warnings
warnings.simplefilter("ignore")

# Initialize google chrome browser
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless=new')
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome('chromedriver', options = chrome_options)

# Create a folder for storing data
root_dir = 'Stock_price/cafef_data'
os.makedirs(root_dir, exist_ok = True)

# Access to the page
main_url = 'https://s.cafef.vn/Lich-su-giao-dich-FPT-1.chn'
driver.get(main_url)

n_page = 200  # 4000 data_points
ngay = []
gia_dieu_chinh = []
gia_dong_cua = []
gia_mo_cua = []
gia_cao_nhat = []
gia_thap_nhat = []

for i in range(n_page):

    date_list = driver.find_elements(By.CLASS_NAME, 'Item_DateItem')
    for element in date_list:
        ngay.append(element.text)

    data_list = driver.find_elements(By.CLASS_NAME, 'Item_Price10')
    count = 0
    for element in data_list:
        if (count % 11 == 0):
            gia_dieu_chinh.append(element.text)
        elif (count % 11 == 1):
            gia_dong_cua.append(element.text)
        elif (count % 11 == 5):
            gia_mo_cua.append(element.text)
        elif (count % 11 == 6):
            gia_cao_nhat.append(element.text)
        elif (count % 11 == 7):
            gia_thap_nhat.append(element.text)

        count += 1

    if i == 0:
        chuyen_trang = driver.find_element(By.XPATH, '/html/body/form/div[3]/div/div[2]/div[2]/div[1]/div[3]/div/div/div[2]/div[2]/div[2]/div/div/div/div/table/tbody/tr/td[2]/a')
        chuyen_trang.click()
        sleep(2)
    elif 1 <= i <= 18:
        chuyen_trang = driver.find_element(By.XPATH, f'/html/body/form/div[3]/div/div[2]/div[2]/div[1]/div[3]/div/div/div[2]/div[2]/div[2]/div/div/div/div/table/tbody/tr/td[{i + 3}]/a')
        chuyen_trang.click()
        sleep(2)
    else:
        chuyen_trang = driver.find_element(By.XPATH, '/html/body/form/div[3]/div/div[2]/div[2]/div[1]/div[3]/div/div/div[2]/div[2]/div[2]/div/div/div/div/table/tbody/tr/td[13]/a')
        chuyen_trang.click()
        sleep(2)

dict_columns = {
    'Ngay':ngay,
    'Gia_dieu_chinh':gia_dieu_chinh,
    'Gia_dong_cua':gia_dong_cua,
    'Gia_mo_cua':gia_mo_cua,
    'Gia_cao_nhat':gia_cao_nhat,
    'Gia_thap_nhat':gia_thap_nhat,
    }

df = pd.DataFrame(dict_columns)
print(df.head())
print("-"*80)
print(df.tail())

df.to_csv('cafef_data/FPT_data.csv')

