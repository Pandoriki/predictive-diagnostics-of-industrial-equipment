import React, { useState } from 'react';
import { Equipment, Status, Theme } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { AlertTriangleIcon, BellIcon, CheckCircleIcon } from './icons';

interface EngineerDashboardViewProps {
  equipment: Equipment;
  theme: Theme;
}

const statusConfig = {
  [Status.Normal]: { color: 'green-400', icon: CheckCircleIcon, ring: 'green-500/50' },
  [Status.Warning]: { color: 'yellow-400', icon: BellIcon, ring: 'yellow-500/50' },
  [Status.Critical]: { color: 'red-500', icon: AlertTriangleIcon, ring: 'red-500/50' },
};

const GlassPanel: React.FC<{children: React.ReactNode, className?: string}> = ({ children, className }) => (
    <div className={`relative p-6 rounded-3xl bg-white/20 dark:bg-black/30 backdrop-blur-lg border border-black/10 dark:border-white/10 shadow-2xl shadow-black/30 ${className}`}>
        <div className="absolute top-0 left-1/4 w-1/2 h-1/2 bg-gradient-to-br from-white/5 to-transparent opacity-50 -translate-x-1/4 -translate-y-1/4 blur-3xl"></div>
        <div className="relative z-10 h-full">{children}</div>
    </div>
);

const ChartTooltip: React.FC<any> = ({ active, payload, label, theme }) => {
  if (active && payload && payload.length) {
    return (
      <div className="p-3 rounded-xl dark:bg-slate-800/80 bg-white/80 backdrop-blur-md border dark:border-slate-600 border-slate-300 shadow-lg">
        <p className={`label text-sm dark:text-gray-300 text-slate-700`}>{`Цикл: ${label}`}</p>
        {payload.map((p: any, index: number) => ( // Note: payload color is passed by Recharts, no theme change needed here
            <p key={index} style={{ color: p.color }} className="text-sm font-medium">{`${p.name}: ${p.value.toFixed(2)}`}</p>
        ))}
      </div>
    );
  }
  return null;
};

const EngineerDashboardView: React.FC<EngineerDashboardViewProps> = ({ equipment, theme }) => {
  const { color, icon: Icon } = statusConfig[equipment.status];
  const [rulTimeRange, setRulTimeRange] = useState<number>(100);

  const filteredRulHistory = equipment.rulHistory.slice(0, rulTimeRange);
  
  const axisColor = theme === Theme.Dark ? '#9ca3af' : '#475569';
  const gridColor = theme === Theme.Dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      
      {/* Left Column */}
      <div className="lg:col-span-1 space-y-6">
        <GlassPanel className="flex flex-col items-center justify-center text-center">
            <div className={`relative w-48 h-48 flex items-center justify-center rounded-full bg-gradient-to-br dark:from-gray-800 dark:to-gray-900 from-gray-200 to-gray-100 ring-8 ring-${color.replace('-400','')}/30`}>
                <div className="absolute inset-0 border-4 border-slate-400 dark:border-gray-700 rounded-full"></div>
                <div style={{ transform: `rotate(${(1 - equipment.rul/100) * -360}deg)`}} className="absolute inset-0 transition-transform duration-1000">
                     <div className={`absolute top-1/2 -mt-1 -ml-1 h-2 w-2 rounded-full bg-${color} shadow-lg shadow-${color}`}></div>
                </div>
                <Icon className={`w-16 h-16 text-${color}`} />
            </div>
            <h2 className="mt-4 text-2xl font-bold text-slate-800 dark:text-white">ПРОГНОЗ RUL: {equipment.rul} {equipment.rulUnit}</h2>
            <p className={`mt-1 font-semibold text-${color}`}>{equipment.status}</p>
        </GlassPanel>

        <GlassPanel>
            <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-2">Детальная Диагностика</h3>
            <div className="space-y-3 text-slate-600 dark:text-gray-300 text-sm">
                <p><strong className="text-slate-700 dark:text-gray-200">ID:</strong> {equipment.id} {equipment.name}</p>
                <p><strong className="text-slate-700 dark:text-gray-200">Тип:</strong> {equipment.type}</p>
                <p><strong className="text-slate-700 dark:text-gray-200">Модель:</strong> {equipment.model}</p>
                <p><strong className="text-slate-700 dark:text-gray-200">Последнее Предупреждение:</strong> {equipment.lastWarning}</p>
                <div className="pt-2 mt-2 border-t border-black/10 dark:border-white/10">
                    <p className="font-semibold text-slate-800 dark:text-gray-100 mb-1">Предполагаемая Причина Деградации:</p>
                    <p className="text-cyan-600 dark:text-cyan-300">{equipment.degradationReason}</p>
                </div>
            </div>
        </GlassPanel>
      </div>

      {/* Right Column */}
      <div className="lg:col-span-2 space-y-6">
        <GlassPanel className="h-[400px]">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Динамика Прогноза RUL</h3>
                <div className="flex items-center space-x-2">
                    {[100, 500, equipment.rulHistory.length].map(range => (
                        <button 
                            key={range}
                            onClick={() => setRulTimeRange(range)}
                            className={`px-3 py-1 text-xs rounded-full transition-colors ${rulTimeRange === range ? 'bg-cyan-500/30 text-cyan-800 dark:text-cyan-300' : 'bg-black/10 dark:bg-white/10 text-slate-600 dark:text-gray-400 hover:bg-black/20 dark:hover:bg-white/20'}`}
                        >
                            {range === equipment.rulHistory.length ? 'Все' : `Последние ${range}`}
                        </button>
                    ))}
                </div>
            </div>
            <ResponsiveContainer width="100%" height="90%">
                <LineChart data={filteredRulHistory} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis dataKey="cycle" stroke={axisColor} fontSize={12} label={{ value: 'Циклы', position: 'insideBottom', offset: -5, fill: axisColor }} />
                    <YAxis stroke={axisColor} fontSize={12} />
                    <Tooltip content={<ChartTooltip theme={theme} />} />
                    <ReferenceLine y={20} label={{ value: 'Критич.', fill: '#ef4444', fontSize: 12, position: 'insideTopLeft' }} stroke="#ef4444" strokeDasharray="4 4" />
                    <ReferenceLine y={50} label={{ value: 'Обслуж.', fill: '#f59e0b', fontSize: 12, position: 'insideTopLeft' }} stroke="#f59e0b" strokeDasharray="4 4" />
                    <Line type="monotone" dataKey="predicted" name="Прогноз" stroke="#06b6d4" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="actual" name="Псевдо-Истинный" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </GlassPanel>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* FIX: Explicitly type 'sensor' as TypeScript may not correctly infer it from Object.values. */}
            {Object.values(equipment.sensors).map((sensor: Equipment['sensors'][string], index) => (
                <GlassPanel key={index} className="h-[300px]">
                    <h4 className="text-md font-bold text-slate-800 dark:text-white mb-2">{sensor.name}</h4>
                     <ResponsiveContainer width="100%" height="90%">
                        <LineChart data={sensor.data} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                            <XAxis dataKey="cycle" stroke={axisColor} fontSize={12} />
                            <YAxis stroke={axisColor} fontSize={12} domain={['dataMin - 1', 'dataMax + 1']}/>
                            <Tooltip content={<ChartTooltip theme={theme} />}/>
                            <Line type="monotone" dataKey="value" name={sensor.name} stroke="#34d399" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </GlassPanel>
            ))}
        </div>
      </div>
    </div>
  );
};

export default EngineerDashboardView;