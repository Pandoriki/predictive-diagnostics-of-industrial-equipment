// frontend/api/apiService.ts

// Импортируем типы, которые описывают ответы от нашего API
import { BackendStatusSummary, HistoryData } from '../types';

// Базовый URL для всех запросов. Берется из переменных окружения.
const API_BASE_URL = '/api';

/**
 * Запрашивает краткую сводку по статусу всего оборудования.
 * @returns {Promise<BackendStatusSummary[]>} Промис, который разрешается в массив объектов статуса.
 */
export const fetchStatusSummary = async (): Promise<BackendStatusSummary[]> => {
  console.log(`[API] Запрос данных с: ${API_BASE_URL}/status_summary`);
  
  const response = await fetch(`${API_BASE_URL}/status_summary`);
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error(`[API] Ошибка запроса сводки: ${response.status}`, errorText);
    throw new Error(`Ошибка сети при запросе сводки: ${response.status}`);
  }

  return await response.json() as BackendStatusSummary[];
};


/**
 * Запрашивает детальную историю для одного конкретного оборудования.
 * @param {number} unitId - Числовой ID оборудования.
 * @returns {Promise<BackendHistoryResponse>} Промис, который разрешается в объект с историей.
 */
export const fetchEquipmentHistory = async (unitId: number): Promise<HistoryData> => {
    console.log(`[API] Запрос истории для оборудования с ID: ${unitId}`);
    
    const response = await fetch(`${API_BASE_URL}/history/${unitId}`);

    if (!response.ok) {
        const errorText = await response.text();
        console.error(`[API] Ошибка запроса истории для ID ${unitId}:`, errorText);
        throw new Error(`Ошибка сети при запросе истории: ${response.status}`);
    }

    return await response.json() as HistoryData;
};

export const fetchHistoryData = async (unitId: number): Promise<HistoryData> => {
  const cleanUnitId = String(unitId).replace('#', '');
  const response = await fetch(`${API_BASE_URL}/history/${cleanUnitId}`);
  if (!response.ok) {
    throw new Error(`Не удалось загрузить историю для оборудования #${cleanUnitId}`);
  }
  return response.json();
};