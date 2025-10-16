
export enum Status {
  Normal = 'НОРМАЛЬНО',
  Warning = 'ТРЕБУЕТ ОБСЛУЖИВАНИЯ',
  Critical = 'КРИТИЧЕСКОЕ СОСТОЯНИЕ',
}

export enum Theme {
  Light = 'light',
  Dark = 'dark',
}

export enum View {
  List = 'list',
  Engineer = 'engineer',
  Guard = 'guard',
}

export interface SensorDataPoint {
  cycle: number;
  value: number;
}

export interface RulDataPoint {
  cycle: number;
  predicted: number;
  actual: number;
}

export interface Equipment {
  id: string;
  name: string;
  status: Status;
  rul: number;
  rulUnit: 'циклов' | 'дн.';
  type: string;
  model: string;
  lastWarning: string;
  degradationReason: string;
  rulHistory: RulDataPoint[];
  sensors: {
    [key: string]: {
      name: string;
      data: SensorDataPoint[];
    };
  };
}
