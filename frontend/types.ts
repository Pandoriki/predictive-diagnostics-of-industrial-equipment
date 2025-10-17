
export interface SensorDataPoint {
  cycle: number;
  value: number;
}

export interface RulDataPoint {
  cycle: number;
  predicted: number;
  actual: number;
}
// frontend/types.ts

export enum Theme {
  Light = 'light',
  Dark = 'dark',
}

export enum View {
  List = 'list',
  Engineer = 'engineer',
  Guard = 'guard',
}

export enum Status {
  Normal = 'Нормально',
  Warning = 'Предупреждение',
  Critical = 'Критическое',
}

// Describe la estructura de datos que usa la UI.
// Es importante que incluya `last_updated`.
export interface Equipment {
  id: string;
  name: string;
  status: Status;
  rul: number;
  rulUnit: string;
  last_updated: string; // <-- Campo crucial
  type: string;
  model: string;
  lastWarning: string;
  degradationReason: string;
  rulHistory: any[];
  sensors: any;
}

// Describe la respuesta EXACTA que viene de la API.
export interface BackendStatusSummary {
  unit_id: number;
  current_rul: number;
  status_ru: string;
  status_code: 'normal' | 'warning' | 'critical';
  status_color: 'зеленый' | 'желтый' | 'красный';
  last_updated: string;
}

export interface HistoryData {
  unit_id: number;
  time_in_cycles: number[];
  rul_history: number[];
  sensor_data: {
    [sensorName: string]: number[];
  };
}

export interface ChartDataPoint {
  cycle: number;
  [key: string]: number; // Для RUL, сенсоров и т.д.
}