import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import os
import re
from datetime import datetime

# ===================================================================
# НАСТРОЙКА СТРАНИЦЫ
# ===================================================================
st.set_page_config(
    page_title="AutoPrice Expert",
    page_icon="🚗",
    layout="wide"
)

# ===================================================================
# МАППИНГИ ДЛЯ ПРЕОБРАЗОВАНИЯ ЗНАЧЕНИЙ
# ===================================================================
# Маппинг из русских значений в английские/технические
VALUE_MAPPINGS = {
    'vehicleTransmission': {
        'автоматическая': 'AUTOMATIC',
        'механическая': 'MECHANICAL',
        'робот': 'ROBOT',
        'вариатор': 'VARIATOR'
    },
    'Руль': {
        'Левый': 'LEFT',
        'Правый': 'RIGHT'
    },
    'ПТС': {
        'Оригинал': 'ORIGINAL',
        'Дубликат': 'DUPLICATE'
    },
    'color': {
        'черный': '040001',
        'белый': 'FFFFFF',
        'серебристый': 'C0C0C0',
        'серый': '808080',
        'синий': '0000FF',
        'красный': 'FF0000',
        'зеленый': '008000',
        'коричневый': 'A52A2A',
        'желтый': 'FFFF00',
        'оранжевый': 'FFA500',
        'фиолетовый': '800080',
        'голубой': '00FFFF',
        'розовый': 'FFC0CB',
        'бордовый': '800000',
        'бежевый': 'F5F5DC',
        'золотой': 'FFD700',
        'бирюзовый': '40E0D0'
    }
}


# ===================================================================
# ФУНКЦИИ ДЛЯ МОДЕЛИ
# ===================================================================
@st.cache_resource
def load_model_and_encoders():
    """Загружаем модель и кодировщики"""
    try:
        # Загрузка модели
        model_path = 'models/lightgbm_car_price_model.txt'
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            return None

        loaded_model = lgb.Booster(model_file=model_path)

        # Загрузка label encoders
        encoders_path = 'models/label_encoders.pkl'
        if not os.path.exists(encoders_path):
            st.error(f"Файл кодировщиков не найден: {encoders_path}")
            return None

        with open(encoders_path, 'rb') as f:
            loaded_encoders = pickle.load(f)

        # Загрузка названий признаков
        feature_names = None
        features_path = 'models/feature_names.pkl'
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            # Если файла нет, создаем список признаков из модели или кодировщиков
            print("Файл feature_names.pkl не найден. Используем признаки из кодировщиков.")
            feature_names = list(loaded_encoders.keys()) + [
                'productionDate', 'mileage', 'enginePower', 'engineDisplacement',
                'numberOfDoors', 'Владельцы', 'engineDisplacement_num',
                'description_length', 'start_year', 'start_month', 'start_day'
            ]

        return {
            'model': loaded_model,
            'encoders': loaded_encoders,
            'feature_names': feature_names
        }

    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None


def prepare_features_for_model(car_features, model_data):
    """
    Подготавливает и преобразует пользовательские данные в формат модели
    """
    try:
        # Создаем копию входных данных
        prepared_features = car_features.copy()

        # Преобразуем значения в нужный формат
        for field, mapping in VALUE_MAPPINGS.items():
            if field in prepared_features and prepared_features[field] in mapping:
                prepared_features[field] = mapping[prepared_features[field]]

        # Генерируем недостающие поля
        # engineDisplacement_num (числовая версия объема двигателя)
        if 'engineDisplacement' in prepared_features:
            try:
                engine_value = str(prepared_features['engineDisplacement'])
                # Извлекаем число из строки (например, "2.0" из "2.0 LTR")
                num_match = re.search(r'(\d+\.?\d*)', engine_value)
                if num_match:
                    prepared_features['engineDisplacement_num'] = float(num_match.group(1))
                else:
                    prepared_features['engineDisplacement_num'] = float(engine_value)
            except:
                prepared_features['engineDisplacement_num'] = 2.0

        # modelDate (год модели - обычно совпадает с годом выпуска)
        if 'productionDate' in prepared_features:
            prepared_features['modelDate'] = prepared_features['productionDate']

        # name (формируем из бренда и модели)
        brand = prepared_features.get('brand', '')
        model = prepared_features.get('model', '')
        if brand and model:
            prepared_features['name'] = f"{brand} {model}"
        else:
            prepared_features['name'] = f"+ {prepared_features.get('engineDisplacement', 1.6)} AT"

        # vehicleConfiguration (формируем автоматически)
        body_type = prepared_features.get('bodyType', '').replace(' ', '_').upper()
        transmission = prepared_features.get('vehicleTransmission', 'AUTOMATIC')
        engine = prepared_features.get('engineDisplacement', 1.6)
        prepared_features['vehicleConfiguration'] = f"{body_type}_{transmission}_{engine}"

        # Дефолтные значения для остальных полей
        defaults = {
            'color': '040001',  # черный по умолчанию
            'Комплектация': "{'id': '0', 'name': ''}",
            'Владение': "{'year': 1977, 'month': 12}",
            'model': brand[:3] if brand else 'UNK',
            'description_length': 150,
            'start_year': 2024,
            'start_month': 1,
            'start_day': 1
        }

        for key, value in defaults.items():
            if key not in prepared_features:
                prepared_features[key] = value

        return prepared_features

    except Exception as e:
        st.error(f"Ошибка подготовки данных: {str(e)}")
        return None


def predict_car_price(car_features, model_data):
    """
    Предсказывает цену автомобиля на основе характеристик
    """
    try:
        model = model_data['model']
        loaded_encoders = model_data['encoders']
        feature_names = model_data['feature_names']

        # Подготавливаем данные
        prepared_features = prepare_features_for_model(car_features, model_data)
        if prepared_features is None:
            return None

        # Создаем DataFrame
        input_df = pd.DataFrame([prepared_features])

        # Применяем кодирование к категориальным признакам
        for col in input_df.columns:
            if col in loaded_encoders:
                try:
                    # Преобразуем значение в строку
                    input_value = str(input_df[col].iloc[0])
                    known_categories = set(loaded_encoders[col].classes_)

                    if input_value not in known_categories:
                        # Заменяем на самое частое значение
                        most_frequent = loaded_encoders[col].classes_[0]
                        print(
                            f"⚠️ Замена неизвестной категории '{input_value}' на '{most_frequent}' в признаке {col}")
                        input_value = most_frequent

                    input_df[col] = loaded_encoders[col].transform([input_value])[0]
                except Exception as e:
                    st.error(f"Ошибка кодирования признака {col}: {str(e)}")
                    return None

        # Добавляем отсутствующие признаки (заполняем 0)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Упорядочиваем столбцы как при обучении
        input_df = input_df[feature_names]

        # Проверяем типы данных
        for col in input_df.select_dtypes(include=['object']).columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                st.error(f"Не удалось преобразовать признак {col} в числовой тип")
                return None

        # Предсказание
        predicted_price = model.predict(input_df)[0]

        # Округляем до тысяч
        predicted_price = round(predicted_price, -3)

        return predicted_price

    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {str(e)}")
        return None


# ===================================================================
# ДАННЫЕ ДЛЯ АВТОДОПОЛНЕНИЯ
# ===================================================================
car_data = {
    'brand': [
        "AUDI", "BMW", "CADILLAC", "CHERY", "CHEVROLET", "CHRYSLER", "CITROEN", "DAEWOO",
        "DODGE", "FORD", "GEELY", "GREAT_WALL", "HONDA", "HYUNDAI", "INFINITI", "JAGUAR",
        "JEEP", "KIA", "LAND_ROVER", "LEXUS", "MAZDA", "MERCEDES", "MINI", "MITSUBISHI",
        "NISSAN", "OPEL", "PEUGEOT", "PORSCHE", "RENAULT", "SKODA", "SSANG_YONG", "SUBARU",
        "SUZUKI", "TOYOTA", "VOLKSWAGEN", "VOLVO"
    ],
    'bodyType': [
        "Внедорожник 3 дв.", "Внедорожник 5 дв.", "Кабриолет", "Компактвэн", "Купе",
        "Лимузин", "Лифтбек", "Микровэн", "Минивэн", "Пикап", "Родстер", "Седан",
        "Тарга", "Универсал 5 дв.", "Фастбек", "Фургон", "Хэтчбек 3 дв.", "Хэтчбек 5 дв."
    ],
    'fuelType': ["бензин", "газ", "гибрид", "дизель", "универсал", "электро"],
    'vehicleTransmission': ["автоматическая", "механическая", "робот", "вариатор"],
    'Привод': ["задний", "передний", "полный"],
    'ПТС': ["Оригинал", "Дубликат"],
    'Руль': ["Левый", "Правый"],
    'color': list(VALUE_MAPPINGS['color'].keys()),
    'numberOfDoors': [0, 2, 3, 4, 5],
    'Владельцы': [1, 2, 3]
}

brand_models = {
    "AUDI": ["A3", "A4", "A6", "Q5", "Q7"],
    "BMW": ["3 Series", "5 Series", "X5", "X3"],
    "TOYOTA": ["Camry", "Corolla", "RAV4", "Land Cruiser"],
    "MERCEDES": ["C-Class", "E-Class", "GLC", "S-Class"],
    "VOLKSWAGEN": ["Golf", "Passat", "Tiguan", "Polo"],
    "HYUNDAI": ["Solaris", "Creta", "Tucson", "Santa Fe"],
    "KIA": ["Rio", "Sportage", "Optima", "Sorento"],
    "NISSAN": ["Qashqai", "X-Trail", "Teana", "Murano"],
    "MAZDA": ["CX-5", "6", "3", "CX-9"],
    "LEXUS": ["RX", "NX", "ES", "LX"]
}


# ===================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ИНТЕРФЕЙСА
# ===================================================================
def create_car_features_dict(brand, model_name, year, mileage, engine_power,
                             engine_volume, fuel_type, transmission, body_type,
                             drive, pts, wheel, color, doors, owners):
    """Создает словарь с характеристиками для модели"""
    return {
        'brand': str(brand).upper() if brand else '',
        'model': str(model_name) if model_name else '',
        'productionDate': int(year),
        'mileage': int(mileage),
        'enginePower': float(engine_power),
        'engineDisplacement': float(engine_volume),
        'fuelType': str(fuel_type),
        'vehicleTransmission': str(transmission),
        'bodyType': str(body_type),
        'Привод': str(drive),
        'ПТС': str(pts),
        'Руль': str(wheel),
        'color': str(color),
        'numberOfDoors': int(doors),
        'Владельцы': int(owners)
    }


# ===================================================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ===================================================================
def main():
    st.title("🚗 AutoPrice Expert")
    st.markdown("---")

    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        model_data = load_model_and_encoders()

    if not model_data:
        st.error("""
        ⚠️ Не удалось загрузить модель. Убедитесь, что в папке models/ есть следующие файлы:
        - `lightgbm_car_price_model.txt`
        - `label_encoders.pkl`
        - `feature_names.pkl`
        """)
        st.stop()

    # Две колонки
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📝 Характеристики автомобиля")

        # Основная информация
        st.markdown("#### 🏷️ Основная информация")

        # Марка и модель
        col1, col2 = st.columns(2)
        with col1:
            brand = st.selectbox(
                "Марка автомобиля *",
                options=["Выберите марку"] + car_data['brand'],
                help="Выберите марку автомобиля"
            )

        with col2:
            # Модель - зависит от выбранной марки
            model_options = ["Выберите модель"]
            if brand and brand != "Выберите марку":
                model_options = brand_models.get(brand, ["Выберите модель"])

            model_name = st.selectbox(
                "Модель *",
                options=model_options,
                index=0,
                disabled=(brand == "Выберите марку" or brand == "")
            )

        with st.form("car_form"):


            # Год выпуска и пробег
            col1, col2 = st.columns(2)
            with col1:
                year = st.slider(
                    "Год выпуска *",
                    1990, 2024, 2018,
                    key="year"
                )

            with col2:
                mileage = st.number_input(
                    "Пробег (км) *",
                    0, 1000000, 50000, 1000,
                    key="mileage",
                    format="%d"
                )

            st.markdown("---")

            # Технические характеристики
            st.markdown("#### ⚙️ Технические характеристики")

            col1, col2 = st.columns(2)
            with col1:
                engine_power = st.slider(
                    "Мощность двигателя (л.с.) *",
                    50, 500, 150, 10,
                    key="power"
                )

                engine_volume = st.slider(
                    "Объем двигателя (л) *",
                    0.8, 5.0, 2.0, 0.1,
                    key="volume"
                )

            with col2:
                fuel_type = st.selectbox(
                    "Тип топлива *",
                    options=car_data['fuelType'],
                    index=0
                )

                transmission = st.selectbox(
                    "Коробка передач *",
                    options=car_data['vehicleTransmission'],
                    index=0
                )

            st.markdown("---")

            # Внешний вид
            st.markdown("#### 🎨 Внешний вид")

            col1, col2 = st.columns(2)
            with col1:
                body_type = st.selectbox(
                    "Тип кузова *",
                    options=car_data['bodyType'],
                    index=11  # Седан по умолчанию
                )

                color = st.selectbox(
                    "Цвет *",
                    options=car_data['color'],
                    index=0  # Черный по умолчанию
                )

            with col2:
                drive = st.selectbox(
                    "Привод *",
                    options=car_data['Привод'],
                    index=1  # Передний по умолчанию
                )

                doors = st.selectbox(
                    "Количество дверей *",
                    options=car_data['numberOfDoors'],
                    index=3,  # 4 двери по умолчанию
                    key="doors"
                )

            st.markdown("---")

            # Документы и владельцы
            st.markdown("#### 📋 Документы и история")

            col1, col2, col3 = st.columns(3)
            with col1:
                pts = st.selectbox(
                    "ПТС *",
                    options=car_data['ПТС'],
                    index=0  # Оригинал по умолчанию
                )

            with col2:
                wheel = st.selectbox(
                    "Руль *",
                    options=car_data['Руль'],
                    index=0  # Левый по умолчанию
                )

            with col3:
                owners = st.selectbox(
                    "Количество владельцев *",
                    options=car_data['Владельцы'],
                    index=1,  # 2 владельца по умолчанию
                    key="owners"
                )

            # Кнопка расчета
            calculate_button = st.form_submit_button(
                "🎯 РАССЧИТАТЬ СТОИМОСТЬ",
                use_container_width=True,
                type="primary",
                disabled=(brand == "Выберите марку" or
                          model_name == "Выберите модель")
            )

    # Обработка формы
    if calculate_button:
        with col_right:
            st.subheader("📊 Результат оценки")

            # Собираем данные
            car_features = create_car_features_dict(
                brand, model_name, year, mileage, engine_power,
                engine_volume, fuel_type, transmission, body_type,
                drive, pts, wheel, color, doors, owners
            )

            # Проверяем обязательные поля
            required_fields = ['brand', 'productionDate', 'mileage', 'enginePower']
            missing_fields = [field for field in required_fields
                              if not car_features.get(field)]

            if missing_fields:
                st.error(f"Заполните обязательные поля: {', '.join(missing_fields)}")
            else:
                with st.spinner('🤖 Модель анализирует данные...'):
                    # ПРЕДСКАЗАНИЕ МОДЕЛЬЮ
                    predicted_price = predict_car_price(car_features, model_data)

                    if predicted_price:
                        st.success("✅ Оценка завершена!")

                        # Отображаем результат
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 25px; 
                                    border-radius: 15px; 
                                    color: white;
                                    text-align: center;
                                    margin: 20px 0;">
                            <h1 style="margin: 0; font-size: 36px;">💰 {predicted_price:,.0f} руб.</h1>
                            <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">
                                Примерная рыночная стоимость
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Доверительный интервал
                        mae = 135699
                        lower_bound = max(0, predicted_price - mae)
                        upper_bound = predicted_price + mae

                        st.info(f"""
                        **📈 Доверительный интервал:**

                        **{lower_bound:,.0f} - {upper_bound:,.0f} руб.**

                        *Средняя ошибка модели: {mae:,.0f} руб.*
                        """)

                        # Информация о точности
                        with st.expander("ℹ️ О точности модели"):
                            st.markdown("""
                            **Метрики модели LightGBM:**
                            - Средняя абсолютная ошибка (MAE): ~136,000 руб.
                            - Точность в пределах 15%: ~70%
                            - Модель объясняет ~89% дисперсии цен

                            *Оценка основана на данных о продажах подержанных автомобилей.*
                            """)
                    else:
                        st.error("❌ Не удалось получить оценку.")


# ===================================================================
# ЗАПУСК
# ===================================================================
if __name__ == "__main__":
    main()