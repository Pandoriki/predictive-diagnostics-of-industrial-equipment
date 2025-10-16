
import { Equipment, Status } from './types';

const generateRulHistory = (startRul: number, length: number, degradationRate: number) => {
  const history = [];
  for (let i = 0; i < length; i++) {
    const predicted = startRul - i * degradationRate * (1 + (Math.random() - 0.5) * 0.1);
    const actual = startRul - i * degradationRate * (1 + (Math.random() - 0.5) * 0.2);
    history.push({ cycle: i + 1, predicted: Math.max(0, predicted), actual: Math.max(0, actual) });
  }
  return history.reverse();
};

const generateSensorData = (base: number, length: number, trend: number, noise: number) => {
    const data = [];
    for (let i = 0; i < length; i++) {
        data.push({ cycle: i + 1, value: base + i * trend + (Math.random() - 0.5) * noise });
    }
    return data;
};

export const mockEquipmentData: Equipment[] = [
  {
    id: '#1004',
    name: 'Турбина А',
    status: Status.Critical,
    rul: 5,
    rulUnit: 'дн.',
    type: 'Газотурбинный двигатель',
    model: 'GT-3000',
    lastWarning: 'Высокая вибрация вала',
    degradationReason: 'Увеличение высокочастотной вибрации на ~30% в диапазоне 20-50 Гц указывает на критический износ подшипника вала №3.',
    rulHistory: generateRulHistory(100, 200, 0.5),
    sensors: {
        vibrationX: { name: "Вибрация по оси X (Отфильтрованная)", data: generateSensorData(1.5, 200, 0.02, 0.5) },
        temperature: { name: "Температура Двигателя", data: generateSensorData(850, 200, 0.5, 20) },
        pressure: { name: "Давление Масла", data: generateSensorData(5.2, 200, -0.005, 0.2) },
    }
  },
  {
    id: '#2501',
    name: 'Компрессор B-12',
    status: Status.Warning,
    rul: 18,
    rulUnit: 'циклов',
    type: 'Центробежный компрессор',
    model: 'CC-120',
    lastWarning: 'Нестабильное давление на выходе',
    degradationReason: 'Периодические скачки давления масла в системе смазки. Рекомендуется проверка масляного насоса и фильтров.',
    rulHistory: generateRulHistory(200, 200, 0.9),
    sensors: {
        vibrationX: { name: "Вибрация по оси X (Отфильтрованная)", data: generateSensorData(0.8, 200, 0.005, 0.2) },
        temperature: { name: "Температура Двигателя", data: generateSensorData(120, 200, 0.2, 5) },
        pressure: { name: "Давление Масла", data: generateSensorData(8.0, 200, -0.01, 0.8) },
    }
  },
  {
    id: '#007',
    name: 'Насос охлаждения',
    status: Status.Normal,
    rul: 35,
    rulUnit: 'дн.',
    type: 'Водяной насос',
    model: 'WP-500',
    lastWarning: 'Нет',
    degradationReason: 'Все параметры в пределах нормы. Признаков ускоренной деградации не обнаружено.',
    rulHistory: generateRulHistory(50, 50, 0.3),
    sensors: {
        vibrationX: { name: "Вибрация по оси X (Отфильтрованная)", data: generateSensorData(0.3, 50, 0, 0.1) },
        temperature: { name: "Температура Двигателя", data: generateSensorData(65, 50, 0.01, 2) },
        pressure: { name: "Давление Масла", data: generateSensorData(6.1, 50, 0, 0.1) },
    }
  },
  {
    id: '#3145',
    name: 'Генератор G-3',
    status: Status.Normal,
    rul: 92,
    rulUnit: 'дн.',
    type: 'Электрогенератор',
    model: 'EG-15MW',
    lastWarning: 'Нет',
    degradationReason: 'Стабильная работа. Незначительные флуктуации температуры в пределах допустимых норм.',
    rulHistory: generateRulHistory(100, 8, 0.1),
    sensors: {
        vibrationX: { name: "Вибрация по оси X (Отфильтрованная)", data: generateSensorData(0.2, 8, 0, 0.05) },
        temperature: { name: "Температура Двигателя", data: generateSensorData(75, 8, 0.05, 1.5) },
        pressure: { name: "Давление Масла", data: generateSensorData(7.5, 8, 0, 0.05) },
    }
  },
  {
    id: '#5899',
    name: 'Робот-манипулятор R-5',
    status: Status.Warning,
    rul: 40,
    rulUnit: 'циклов',
    type: 'Роботизированная рука',
    model: 'Kuka KR 210',
    lastWarning: 'Повышенный люфт в суставе J3',
    degradationReason: 'Анализ вибрационных данных показывает износ редуктора в третьем суставе. Рекомендуется калибровка и возможное обслуживание.',
    rulHistory: generateRulHistory(300, 260, 1),
    sensors: {
        vibrationX: { name: "Вибрация сустава J3", data: generateSensorData(2.1, 260, 0.003, 0.4) },
        temperature: { name: "Температура привода J3", data: generateSensorData(88, 260, 0.05, 4) },
        pressure: { name: "Ток двигателя J3", data: generateSensorData(15.5, 260, 0.01, 1) },
    }
  },
];
