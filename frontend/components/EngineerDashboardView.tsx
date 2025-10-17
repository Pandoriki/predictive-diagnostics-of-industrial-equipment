// frontend/src/components/EngineerDashboardView.tsx (ЗАМЕНИТЕ ПОЛНОСТЬЮ)

import React, { useState, useEffect, useMemo } from 'react';
import { Equipment, Status, Theme, HistoryData, ChartDataPoint } from '../types';
import { fetchHistoryData } from '../api/apiService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { AlertTriangleIcon, BellIcon, CheckCircleIcon, ArrowLeftIcon } from './icons';

// --- ИНТЕРФЕЙСЫ И ДОЧЕРНИЕ КОМПОНЕНТЫ ---

interface EngineerDashboardViewProps {
  initialEquipment: Equipment;
  onClose: () => void;
  theme: Theme;
}

const statusConfig = {
  [Status.Normal]: { color: 'text-green-400', icon: CheckCircleIcon, ring: 'ring-green-500/30' },
  [Status.Warning]: { color: 'text-yellow-400', icon: BellIcon, ring: 'ring-yellow-500/30' },
  [Status.Critical]: { color: 'text-red-400', icon: AlertTriangleIcon, ring: 'ring-red-500/30' },
};

const Card: React.FC<{children: React.ReactNode, className?: string, theme: Theme}> = ({ children, className, theme }) => (
    <div className={`p-6 rounded-3xl border ${className} 
                   ${theme === Theme.Dark ? 'bg-zinc-900 border-zinc-800' : 'bg-white border-zinc-200'}`}>
        {children}
    </div>
);

const ChartTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="p-3 rounded-xl bg-zinc-50/80 dark:bg-zinc-800/80 backdrop-blur-md border border-zinc-300 dark:border-zinc-600 shadow-lg">
        <p className="label text-sm text-zinc-700 dark:text-zinc-300">{`Цикл: ${label}`}</p>
        {payload.map((p: any, index: number) => (
            <p key={index} style={{ color: p.color }} className="text-sm font-semibold">{`${p.name}: ${p.value.toFixed(2)}`}</p>
        ))}
      </div>
    );
  }
  return null;
};


// --- ФУНКЦИЯ-"АДАПТЕР" ДАННЫХ ---
const adaptHistoryDataForCharts = (historyData: HistoryData): { rulChartData: ChartDataPoint[], sensorChartData: { [key: string]: ChartDataPoint[] } } => {
  
  // Новый API уже дает данные в удобном формате, поэтому адаптер становится намного проще.
  const timeAxis = historyData.time_in_cycles;

  const rulChartData: ChartDataPoint[] = timeAxis.map((cycle, index) => ({
    cycle,
    'Прогноз RUL': historyData.rul_history[index],
  }));

  const sensorChartData: { [key: string]: ChartDataPoint[] } = {};
  for (const sensorName in historyData.sensor_data) {
    sensorChartData[sensorName] = timeAxis.map((cycle, index) => ({
      cycle,
      [sensorName]: historyData.sensor_data[sensorName][index],
    }));
  }

  return { rulChartData, sensorChartData };
};


// --- ОСНОВНОЙ КОМПОНЕНТ ---

const EngineerDashboardView: React.FC<EngineerDashboardViewProps> = ({ initialEquipment, onClose, theme }) => {
    const [chartData, setChartData] = useState<{ rulChartData: ChartDataPoint[], sensorChartData: { [key: string]: ChartDataPoint[] } }>({ rulChartData: [], sensorChartData: {} });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [rulTimeRange, setRulTimeRange] = useState<number>(300);

    useEffect(() => {
        const loadDetailedData = async () => {
            const unitId = parseInt(initialEquipment.id.replace('#', ''));
            if (isNaN(unitId)) {
                setError('ID оборудования некорректен.');
                setLoading(false);
                return;
            }
            try {
                setLoading(true);
                const historyResponse = await fetchHistoryData(unitId);
                const adaptedData = adaptHistoryDataForCharts(historyResponse);
                setChartData(adaptedData);
            } catch (err) {
                setError("Не удалось загрузить детальную информацию.");
            } finally {
                setLoading(false);
            }
        };
        loadDetailedData();
    }, [initialEquipment.id]);

    const { color, icon: Icon, ring } = statusConfig[initialEquipment.status];
    const filteredRulHistory = useMemo(() => chartData.rulChartData.slice(-rulTimeRange), [chartData.rulChartData, rulTimeRange]);

    const axisColor = theme === Theme.Dark ? '#a1a1aa' : '#52525b';
    const gridColor = theme === Theme.Dark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';

    if (loading) return <div className="text-center pt-20 text-xl text-zinc-400">Загрузка детальных данных...</div>;
    if (error) return <div className="text-center pt-20 text-xl text-red-500">{error}</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in-up">
      
      <div className="lg:col-span-1 space-y-6">
        <button 
          onClick={onClose}
          className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-2xl text-sm font-semibold transition-colors duration-300
                     ${theme === Theme.Dark ? 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700' : 'bg-zinc-200 text-zinc-600 hover:bg-zinc-300'}`}
        >
          <ArrowLeftIcon className="w-5 h-5" />
          Назад к общему списку
        </button>

        <Card theme={theme} className="flex flex-col items-center text-center">
            <div className={`relative w-40 h-40 flex items-center justify-center rounded-full ring-8 ${ring}
                           ${theme === Theme.Dark ? 'bg-zinc-800' : 'bg-zinc-100'}`}>
                <Icon className={`w-16 h-16 ${color}`} />
            </div>
            <h2 className={`mt-4 text-2xl font-bold ${theme === Theme.Dark ? 'text-zinc-100' : 'text-zinc-800'}`}>
              Прогноз RUL: {Math.floor(initialEquipment.rul)} {initialEquipment.rulUnit}
            </h2>
            <p className={`mt-1 font-semibold ${color}`}>{initialEquipment.status}</p>
        </Card>

        <Card theme={theme}>
            <h3 className={`text-xl font-bold mb-3 ${theme === Theme.Dark ? 'text-zinc-100' : 'text-zinc-800'}`}>Информация об объекте</h3>
            <div className={`space-y-2 text-sm ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>
                <p><strong className={`${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-700'}`}>ID:</strong> {initialEquipment.id} {initialEquipment.name}</p>
                <p><strong className={`${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-700'}`}>Тип:</strong> {initialEquipment.type}</p>
                <p><strong className={`${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-700'}`}>Модель:</strong> {initialEquipment.model}</p>
            </div>
        </Card>
      </div>

      <div className="lg:col-span-2 space-y-6">
        <Card theme={theme} className="h-[400px]">
            <div className="flex justify-between items-center mb-4">
                <h3 className={`text-xl font-bold ${theme === Theme.Dark ? 'text-zinc-100' : 'text-zinc-800'}`}>Динамика прогноза RUL</h3>
                {/* Здесь можно будет добавить кнопки для смены диапазона */}
            </div>
            <ResponsiveContainer width="100%" height="90%">
                <LineChart data={filteredRulHistory} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis dataKey="cycle" stroke={axisColor} fontSize={12} label={{ value: 'Циклы', position: 'insideBottom', offset: -5, fill: axisColor }} />
                    <YAxis stroke={axisColor} fontSize={12} />
                    <Tooltip content={<ChartTooltip />} />
                    <ReferenceLine y={20} label={{ value: 'Критич.', fill: '#ef4444', fontSize: 12, position: 'insideTopLeft' }} stroke="#ef4444" strokeDasharray="4 4" />
                    <ReferenceLine y={50} label={{ value: 'Обслуж.', fill: '#f59e0b', fontSize: 12, position: 'insideTopLeft' }} stroke="#f59e0b" strokeDasharray="4 4" />
                    <Line type="monotone" dataKey="Прогноз RUL" name="Прогноз" stroke="#10b981" strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.keys(chartData.sensorChartData).map((sensorName, index) => (
                <Card key={index} theme={theme} className="h-[300px]">
                    <h4 className={`text-md font-bold mb-2 ${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-800'}`}>{sensorName}</h4>
                     <ResponsiveContainer width="100%" height="90%">
                        <LineChart data={chartData.sensorChartData[sensorName]} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                            <XAxis dataKey="cycle" stroke={axisColor} fontSize={12} />
                            <YAxis stroke={axisColor} fontSize={12} domain={['dataMin - 1', 'dataMax + 1']}/>
                            <Tooltip content={<ChartTooltip />}/>
                            <Line type="monotone" dataKey={sensorName} name={sensorName} stroke="#38bdf8" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </Card>
            ))}
        </div>
      </div>
    </div>
  );
};

export default EngineerDashboardView;