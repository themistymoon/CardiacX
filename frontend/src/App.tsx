import React, { useState, useEffect, useRef, FC, Dispatch, SetStateAction, ElementType } from 'react';
import axios, { AxiosResponse, AxiosError } from 'axios';
import { useTranslation } from 'react-i18next';
import { 
  Upload, Bot, User, Loader, AlertCircle, HeartPulse, Mic, Square, Trash2, Download, 
  Activity, ShieldCheck, LifeBuoy, Volume2, BarChart2, ListChecks,
  TrendingUp,
  Heart, Gauge, ListTree, Wand, FileText, Share2, Clipboard, AlertTriangle,
  MessageCircle, Sparkles, Lightbulb, Salad, Dumbbell, Stethoscope, Siren, BrainCircuit, UserCheck, Crosshair, Gem,
  Sun, Moon
} from 'lucide-react';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend
} from 'recharts';

// --- Utility Functions ---
async function callBackendWithRetry<T>(
  url: string, 
  options: any, 
  maxRetries: number = 5
): Promise<AxiosResponse<T>> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await axios(url, options);
      if (response.status < 500) return response;
      
      // If it's a 502 error and we have retries left, wait and retry
      if (response.status === 502 && i < maxRetries - 1) {
        console.log(`Backend not ready (502), retrying in ${2000 * (i + 1)}ms... (attempt ${i + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1)));
        continue;
      }
      
      throw new Error(`HTTP ${response.status}`);
    } catch (error) {
      const axiosError = error as AxiosError;
      
      // If it's a connection error and we have retries left, wait and retry
      if (axiosError.code === 'ECONNREFUSED' && i < maxRetries - 1) {
        console.log(`Backend not ready (connection refused), retrying in ${2000 * (i + 1)}ms... (attempt ${i + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1)));
        continue;
      }
      
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1)));
    }
  }
  throw new Error('Max retries exceeded');
}

// --- Type Definitions ---
type TabName = 'analysis' | 'assistant';
type ResultView = 'distribution' | 'details';

interface AnalysisResult {
  predicted_condition: string;
  confidence: number;
  medical_info: {
    description: string;
    recommendations?: string[];
    severity?: string;
    urgency?: string;
  };
  probabilities: Record<string, number>;
}

interface ChatMessage {
  role: 'bot' | 'user';
  text: string;
}

// --- Main App Component ---
const App: FC = () => {
  const [activeTab, setActiveTab] = useState<TabName>('analysis');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');
  const { i18n } = useTranslation();

  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    const styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = `
      @keyframes fade-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
      .animate-fade-in { animation: fade-in 0.5s ease-out forwards; }
      @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-50%); } }
      .animate-bounce { animation: bounce 1s infinite; }
    `;
    document.head.appendChild(styleSheet);
    return () => { document.head.removeChild(styleSheet); };
  }, []);

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisResult(result);
  };

  const handleBackToAnalysis = () => {
    setAnalysisResult(null);
  };
  
  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };
  
  const changeLanguage = (lng: 'th' | 'en') => {
    i18n.changeLanguage(lng);
  };

  return (
    <div className="min-h-screen bg-page-bg dark:bg-slate-900 font-sans text-text-primary dark:text-gray-200">
      <Header toggleTheme={toggleTheme} theme={theme} changeLanguage={changeLanguage} />
      <main className="container mx-auto px-4 py-8">
        <div className="w-full max-w-4xl mx-auto">
          <Tabs activeTab={activeTab} setActiveTab={setActiveTab} />
          <div className="mt-6">
            {activeTab === 'analysis' && (
              analysisResult ? (
                <ResultDisplay result={analysisResult} onBack={handleBackToAnalysis} theme={theme} />
              ) : (
                <AnalysisPanel onAnalysisComplete={handleAnalysisComplete} />
              )
            )}
            {activeTab === 'assistant' && <AssistantPanel />}
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

// --- Layout Components (Header, Tabs, Footer) ---
const Header: FC<{ toggleTheme: () => void; theme: string; changeLanguage: (lng: 'th' | 'en') => void; }> = ({ toggleTheme, theme, changeLanguage }) => {
  const { t } = useTranslation();
  return (
    <header className="bg-white dark:bg-slate-800 shadow-md">
      <div className="container mx-auto px-4 py-4 flex flex-col sm:flex-row justify-between sm:items-center gap-4 sm:gap-2">
      <div className="flex items-center gap-3">
          <HeartPulse className="h-8 w-8 sm:h-10 sm:w-10 text-brand-primary" />
        <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-text-primary dark:text-white">CardiacZ</h1>
            <p className="text-xs sm:text-sm text-text-secondary dark:text-gray-400 mt-1">{t('tagline')}</p>
          </div>
        </div>
        <div className="flex items-center gap-4 self-end sm:self-auto">
          <LanguageSwitcher changeLanguage={changeLanguage} />
          <ThemeSwitcher toggleTheme={toggleTheme} theme={theme} />
        </div>
      </div>
    </header>
  );
};

const LanguageSwitcher: FC<{ changeLanguage: (lng: 'th' | 'en') => void }> = ({ changeLanguage }) => {
    const { i18n } = useTranslation();
    const isThai = i18n.language.startsWith('th');

    return (
        <div className="relative w-28 h-8 flex items-center bg-gray-200 dark:bg-slate-700 rounded-full cursor-pointer" onClick={() => changeLanguage(isThai ? 'en' : 'th')}>
            <div className={`absolute left-0 w-14 h-full bg-brand-primary rounded-full transition-transform duration-300 ease-in-out ${isThai ? 'translate-x-0' : 'translate-x-full'}`}></div>
            <div className="w-1/2 h-full flex items-center justify-center z-10">
                <span className={`font-bold text-sm ${isThai ? 'text-white' : 'text-text-secondary dark:text-gray-300'}`}>TH</span>
            </div>
            <div className="w-1/2 h-full flex items-center justify-center z-10">
                <span className={`font-bold text-sm ${!isThai ? 'text-white' : 'text-text-secondary dark:text-gray-300'}`}>EN</span>
      </div>
    </div>
    );
};

const ThemeSwitcher: FC<{ toggleTheme: () => void; theme: string }> = ({ toggleTheme, theme }) => (
  <button onClick={toggleTheme} className="p-2 rounded-full bg-gray-200 dark:bg-slate-700 text-text-secondary dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-slate-600 transition-colors">
    {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
  </button>
);

const Tabs: FC<{ activeTab: TabName; setActiveTab: Dispatch<SetStateAction<TabName>> }> = ({ activeTab, setActiveTab }) => {
  const { t } = useTranslation();
  return (
    <div className="flex border-b border-gray-200 dark:border-gray-700">
    <TabButton name="analysis" activeTab={activeTab} setActiveTab={setActiveTab} icon="🩺">
            {t('Heart Disease Analysis')}
    </TabButton>
    <TabButton name="assistant" activeTab={activeTab} setActiveTab={setActiveTab} icon="🤖">
            {t('AI Assistant')}
    </TabButton>
  </div>
);
};

const TabButton: FC<{ name: TabName; activeTab: TabName; setActiveTab: Dispatch<SetStateAction<TabName>>; icon: string; children: React.ReactNode }> = ({ name, activeTab, setActiveTab, icon, children }) => (
  <button
    onClick={() => setActiveTab(name)}
    className={`flex items-center gap-2 px-4 py-3 -mb-px font-semibold text-base transition-colors duration-200 ${
        activeTab === name ? 'border-b-2 border-brand-primary text-brand-primary' : 'text-text-secondary dark:text-gray-400 hover:text-brand-primary dark:hover:text-brand-primary hover:bg-brand-primary/5 dark:hover:bg-brand-primary/10 rounded-t-md'
    }`}
  >
    <span>{icon}</span> {children}
  </button>
);

const Footer: FC = () => {
  const { t } = useTranslation();
  return (
    <footer className="text-center py-4 text-sm text-text-secondary dark:text-gray-400">
      <p>{t('footer', { year: new Date().getFullYear() })}</p>
      <p className="mt-1">{t('decorated_by')}</p>
  </footer>
);
};

interface AnalysisPanelProps {
  onAnalysisComplete: (result: AnalysisResult) => void;
}
const AnalysisPanel: FC<AnalysisPanelProps> = ({ onAnalysisComplete }) => {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  const handleFileChange = (selectedFile: File | null) => {
    if (selectedFile) {
      setFile(selectedFile);
      setAudioURL(URL.createObjectURL(selectedFile));
      setError('');
    }
  };
  
  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); e.stopPropagation();
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) handleFileChange(droppedFile);
  };
  
  const startRecording = async () => {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices) {
      setError(t("browser_not_supported_error")); return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsRecording(true); setAudioURL(''); setFile(null); audioChunks.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => audioChunks.current.push(event.data);
      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);
        setFile(new File([audioBlob], `recording-${Date.now()}.wav`, { type: 'audio/wav' }));
      };
      recorder.start();
    } catch (err) { setError(t("mic_access_error")); }
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };
    
  const resetAll = () => {
      stopRecording(); setFile(null); setIsLoading(false); setError(''); setAudioURL(''); audioChunks.current = [];
      if(fileInputRef.current) fileInputRef.current.value = '';
  }
  
  const handleDiagnose = async () => {
    if (!file) { setError(t('select_file_error')); return; }
    setIsLoading(true); setError('');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await callBackendWithRetry<AnalysisResult>('/api/analyze', {
        method: 'POST',
        data: formData,
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      onAnalysisComplete(response.data);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || "เกิดข้อผิดพลาดที่ไม่คาดคิด";
      setError(t('analysis_failed', { error: errorMessage }));
    } finally { setIsLoading(false); }
  };
  
  const triggerFileSelect = () => fileInputRef.current?.click();

  return (
    <div className="space-y-6 sm:space-y-8">
      <div className="text-center p-6 sm:p-8 bg-gradient-to-br from-brand-gradient-start to-brand-gradient-end rounded-2xl shadow-xl text-white">
          <HeartPulse className="h-12 w-12 sm:h-16 sm:w-16 mx-auto mb-4 text-white/70" />
          <h2 className="text-3xl sm:text-4xl font-bold">CardiacZ</h2>
          <p className="mt-2 text-base sm:text-lg text-white/80">{t('upload_title')}</p>
      </div>
      <div className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm border border-gray-200 dark:border-slate-700 p-6 rounded-lg shadow-lg">
        <button onClick={isRecording ? stopRecording : startRecording} className={`w-full text-white font-bold py-3 sm:py-4 px-6 rounded-lg transition-all duration-300 ease-in-out shadow-md hover:shadow-xl transform hover:-translate-y-1 ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-brand-primary hover:bg-brand-primary/90'}`}>
          <div className="flex items-center justify-center gap-3 text-lg sm:text-xl">{isRecording ? <Square size={24} /> : <Mic size={24} />}<span>{isRecording ? t('recording_button') : t('record_button')}</span></div>
        </button>
        <div className="mt-4 flex justify-center items-center gap-4">
          <button onClick={resetAll} className="flex items-center gap-2 text-text-secondary dark:text-gray-400 hover:text-text-primary dark:hover:text-white"><Trash2 size={16}/> {t('reset_button')}</button>
          {audioURL && (<a href={audioURL} download={`cardiacz-recording-${Date.now()}.wav`} className="flex items-center gap-2 text-text-secondary dark:text-gray-400 hover:text-text-primary dark:hover:text-white"><Download size={16}/> {t('download_button')}</a>)}
        </div>
      </div>
      <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-lg">
        {audioURL && (<div className="mb-4"><h3 className="font-semibold text-lg mb-2 dark:text-white">{t('your_sound')}</h3><audio controls src={audioURL} className="w-full"></audio></div>)}
        <div className="p-8 bg-page-bg/50 dark:bg-slate-700/50 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 text-center cursor-pointer hover:border-brand-primary dark:hover:border-brand-primary hover:bg-brand-primary/10 dark:hover:bg-brand-primary/20 transition-colors" onDrop={handleFileDrop} onDragOver={(e) => e.preventDefault()} onDragEnter={(e) => e.preventDefault()} onClick={triggerFileSelect}>
          <input type="file" ref={fileInputRef} onChange={(e) => handleFileChange(e.target.files?.[0] || null)} className="hidden" accept="audio/wav, audio/mpeg, audio/mp4, audio/aac, audio/ogg, audio/flac, audio/webm" />
          <Upload className="h-12 w-12 text-gray-400 dark:text-gray-500 mx-auto mb-3" />
          <h3 className="font-semibold text-lg mb-1 dark:text-white">{t('drop_file')}</h3>
          <p className="text-sm text-text-secondary dark:text-gray-400">{t('click_select')}</p>
        </div>
      </div>
      <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-lg">
        <button onClick={handleDiagnose} disabled={!file || isLoading} className="w-full bg-brand-primary text-white font-bold py-3 rounded-md hover:bg-brand-primary/90 disabled:bg-brand-primary/50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all text-lg">
          {isLoading ? <><Loader className="animate-spin" /> {t('analyzing_button')}</> : t('diagnose_button')}
        </button>
        {error && <ErrorMessage message={error} />}
      </div>
          </div>
  );
};

const conditionUi: Record<string, {
    icon: ElementType; bgColor: string; textColor: string; borderColor: string; heartColor: string; confidenceColor: string; iconComponent: ElementType;
    darkBgColor?: string; darkTextColor?: string;
  }> = {
    Murmur: { icon: HeartPulse, bgColor: 'bg-red-50', darkBgColor: 'dark:bg-red-900/40', textColor: 'text-red-800', darkTextColor: 'dark:text-red-200', borderColor: 'border-murmur', heartColor: '#EF4444', confidenceColor: 'text-red-600', iconComponent: HeartPulse },
    Normal: { icon: ShieldCheck, bgColor: 'bg-green-50', darkBgColor: 'dark:bg-green-900/40', textColor: 'text-green-800', darkTextColor: 'dark:text-green-200', borderColor: 'border-normal', heartColor: '#22C55E', confidenceColor: 'text-green-600', iconComponent: ShieldCheck },
    Extrahls: { icon: LifeBuoy, bgColor: 'bg-purple-50', darkBgColor: 'dark:bg-purple-900/40', textColor: 'text-purple-800', darkTextColor: 'dark:text-purple-200', borderColor: 'border-extrahls', heartColor: '#8B5CF6', confidenceColor: 'text-purple-600', iconComponent: LifeBuoy },
    Artifact: { icon: Volume2, bgColor: 'bg-slate-50', darkBgColor: 'dark:bg-slate-700/40', textColor: 'text-slate-800', darkTextColor: 'dark:text-slate-200', borderColor: 'border-artifact', heartColor: '#6B7280', confidenceColor: 'text-slate-600', iconComponent: Volume2 },
    Extrastole: { icon: Activity, bgColor: 'bg-orange-50', darkBgColor: 'dark:bg-orange-900/40', textColor: 'text-orange-800', darkTextColor: 'dark:text-orange-200', borderColor: 'border-extrasystole', heartColor: '#F97316', confidenceColor: 'text-orange-600', iconComponent: Activity },
  };
  
const ResultDisplay: FC<{ result: AnalysisResult; onBack: () => void; theme: string }> = ({ result, onBack, theme }) => {
    const [activeView, setActiveView] = useState<ResultView>('distribution');
    const { t } = useTranslation();
    return (
      <div className="space-y-6 animate-fade-in">
        <ResultHeader />
        <ViewSelector activeView={activeView} setActiveView={setActiveView} />
        <div className="mt-8">
            {activeView === 'distribution' ? (
              <ProbabilityDistributionView result={result} theme={theme} />
            ) : (
              <DetailedReportView result={result} />
            )}
        </div>
        <UnderstandingResult />
        <div className="pt-4 text-center">
            <button onClick={onBack} className="bg-brand-primary hover:bg-brand-primary/90 text-white font-bold py-2 px-8 rounded-full transition-all shadow-md">
              {t('analyze_again')}
            </button>
        </div>
      </div>
    );
};
  
const ResultHeader: FC = () => {
    const { t } = useTranslation();
    return (
        <div className="bg-gradient-to-r from-brand-gradient-start to-brand-gradient-end text-white p-6 sm:p-8 rounded-xl shadow-lg text-center flex items-center justify-center gap-4">
            <BarChart2 className="w-8 h-8 sm:w-12 sm:h-12 opacity-80" />
            <div>
                <h2 className="text-2xl sm:text-3xl font-bold">{t('result_header_title')}</h2>
                <p className="mt-1 text-purple-200 text-sm sm:text-base">{t('result_header_subtitle')}</p>
            </div>
        </div>
    );
}

const ViewSelector: FC<{ activeView: ResultView; setActiveView: Dispatch<SetStateAction<ResultView>> }> = ({ activeView, setActiveView }) => {
    const { t } = useTranslation();
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ViewSelectorButton
                icon={TrendingUp} title={t('view_distribution')}
                isActive={activeView === 'distribution'}
                onClick={() => setActiveView('distribution')}
            />
            <ViewSelectorButton
                icon={ListChecks} title={t('view_report')}
                isActive={activeView === 'details'}
                onClick={() => setActiveView('details')}
            />
        </div>
    );
}

const ViewSelectorButton: FC<{ icon: React.ElementType, title: string, isActive: boolean, onClick: () => void }> = ({ icon: Icon, title, isActive, onClick }) => (
    <button onClick={onClick} className={`p-4 sm:p-6 bg-white dark:bg-slate-800 rounded-lg shadow-md hover:shadow-xl hover:-translate-y-1 transition-all flex items-center gap-4 ${isActive ? 'ring-2 ring-brand-primary' : 'ring-1 ring-gray-200 dark:ring-gray-700'}`}>
        <Icon className={`h-8 w-8 sm:h-10 sm:w-10 ${isActive ? 'text-brand-primary' : 'text-text-secondary dark:text-gray-400'}`} />
        <div>
            <h3 className={`text-lg sm:text-xl font-bold ${isActive ? 'text-brand-primary' : 'text-text-primary dark:text-white'}`}>{title}</h3>
        </div>
      </button>
);

const conditionNames: Record<string, string> = {
    Murmur: 'condition_murmur',
    Normal: 'condition_normal',
    Extrahls: 'condition_extrahls',
    Artifact: 'condition_artifact',
    Extrastole: 'condition_extrastole',
};

const CustomLegend = (props: any) => {
    const { payload } = props;
    const { t } = useTranslation();
    return (
        <div className="flex justify-center items-center gap-4 mt-4 flex-wrap">
            {payload.map((entry: any, index: number) => {
                const name = entry.payload.name;
                const ui = conditionUi[name] || conditionUi.Artifact;
                const Icon = ui.iconComponent;
                return (
                    <div key={`item-${index}`} className="flex items-center gap-2">
                        <Icon style={{ color: entry.color }} className="h-4 w-4" />
                        <span className="text-sm text-text-secondary dark:text-gray-400">{t(conditionNames[name] || name)}</span>
                    </div>
                );
            })}
        </div>
    );
};

const ProbabilityDistributionView: FC<{ result: AnalysisResult; theme: string }> = ({ result, theme }) => {
    const { t } = useTranslation();
    const data = Object.entries(result.probabilities)
        .map(([name, value]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            probability: value * 100,
        }))
        .sort((a, b) => b.probability - a.probability);

    const translatedData = data.map(item => ({...item, translatedName: t(conditionNames[item.name] || item.name)}));

    return (
        <div className="space-y-8">
            <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow-lg">
                <div className="flex items-center gap-3 mb-6">
                    <TrendingUp className="h-8 w-8 text-brand-primary" />
                    <h3 className="text-2xl font-bold text-text-primary dark:text-white">{t('probability_distribution_title')}</h3>
                </div>
                <div className="w-full h-96">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={translatedData} margin={{ top: 5, right: 20, left: 10, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-gray-200 dark:stroke-gray-700"/>
                            <XAxis 
                                dataKey="translatedName"
                                tickLine={false} 
                                axisLine={false} 
                                tick={{ fontSize: 12, fill: theme === 'dark' ? '#D1D5DB' : '#6B7280' }} 
                                label={{ value: t('axis_label_condition'), position: 'insideBottom', offset: -10, fill: theme === 'dark' ? '#9CA3AF' : '#6B7280' }}
                            />
                            <YAxis 
                                unit="%" 
                                domain={[0, 100]} 
                                tickLine={false} 
                                axisLine={false} 
                                tick={{ fontSize: 12, fill: theme === 'dark' ? '#D1D5DB' : '#6B7280' }} 
                                label={{ value: t('axis_label_probability'), angle: -90, position: 'insideLeft', fill: theme === 'dark' ? '#9CA3AF' : '#6B7280', dx: -10 }} 
                            />
                            <Tooltip
                                cursor={{ fill: 'rgba(173, 216, 230, 0.3)' }}
                                contentStyle={{
                                    background: theme === 'dark' ? '#1F2937' : 'white',
                                    border: '1px solid #4B5563',
                                    borderRadius: '0.5rem',
                                    color: theme === 'dark' ? '#F3F4F6' : '#1F2937'
                                }}
                                formatter={(value: number) => [`${value.toFixed(1)}%`, t('tooltip_label_probability')]}
                            />
                            <Legend content={<CustomLegend />} verticalAlign="bottom" wrapperStyle={{ paddingTop: '20px' }}/>
                            <Bar dataKey="probability" radius={[8, 8, 0, 0]}>
                                {data.map(entry => (
                                    <Cell key={`cell-${entry.name}`} fill={conditionUi[entry.name]?.heartColor || '#94a3b8'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {data.map(item => <ProbabilityCard key={item.name} name={item.name} probability={item.probability} />)}
            </div>
    </div>
  );
};

const ProbabilityCard: FC<{ name: string; probability: number }> = ({ name, probability }) => {
    const ui = conditionUi[name] || conditionUi.Artifact;
    const Icon = ui.iconComponent;
    const { t } = useTranslation();

    const getConfidence = (prob: number): { level: string; color: string; textColor: string; } => {
        if (prob > 70) return { level: t('confidence_high'), color: "bg-green-500", textColor: "text-green-700" };
        if (prob > 30) return { level: t('confidence_medium'), color: "bg-amber-400", textColor: "text-amber-700" };
        return { level: t('confidence_low'), color: "bg-red-500", textColor: "text-red-700" };
    };
    
    const confidence = getConfidence(probability);

    return (
        <div className={`p-5 rounded-xl shadow-md flex items-center justify-between transition-all hover:shadow-lg hover:-translate-y-1 ${ui.bgColor} ${ui.darkBgColor} border-l-8 ${ui.borderColor}`}>
            <div className="flex-1">
                <div className="flex items-center gap-3">
                    <Icon className={`h-6 w-6 ${ui.textColor} ${ui.darkTextColor}`} />
                    <h4 className={`text-lg sm:text-xl font-bold ${ui.textColor} ${ui.darkTextColor}`}>{t(conditionNames[name] || name)}</h4>
                </div>
                <p className="text-4xl sm:text-5xl font-bold my-2" style={{ color: ui.heartColor }}>
                    {probability.toFixed(1)}%
                </p>
                <div className="flex items-center gap-2">
                    <span className={`w-3 h-3 rounded-full ${confidence.color}`}></span>
                    <span className={`text-sm font-semibold ${confidence.textColor} dark:text-gray-300`}>
                        {t('confidence_level', { level: confidence.level })}
                    </span>
                </div>
            </div>
            <div className="flex-shrink-0">
                <Icon
                    style={{ color: ui.heartColor }}
                    className="w-20 h-20 opacity-30 dark:opacity-20"
                    strokeWidth={1.5}
                />
            </div>
        </div>
    );
};

const conditionDetails: Record<string, {
    name: string;
    description: string;
    recommendations: string[];
    action: string;
  }> = {
    murmur: {
      name: "condition_murmur",
      description: "condition_murmur_desc",
      recommendations: [
        "murmur_rec_1",
        "murmur_rec_2",
        "murmur_rec_3",
        "murmur_rec_4"
      ],
      action: "murmur_action",
    },
    normal: {
      name: "condition_normal",
      description: "condition_normal_desc",
      recommendations: [
        "normal_rec_1",
        "normal_rec_2",
        "normal_rec_3",
        "normal_rec_4"
      ],
      action: "normal_action",
    },
    extrastole: {
      name: "condition_extrastole",
      description: "condition_extrastole_desc",
      recommendations: [
        "extrastole_rec_1",
        "extrastole_rec_2",
        "extrastole_rec_3",
        "extrastole_rec_4"
      ],
      action: "extrastole_action",
    },
    extrahls: {
      name: "condition_extrahls",
      description: "condition_extrahls_desc",
      recommendations: [
        "extrahls_rec_1",
        "extrahls_rec_2",
        "extrahls_rec_3"
      ],
      action: "extrahls_action",
    },
    artifact: {
      name: "condition_artifact",
      description: "condition_artifact_desc",
      recommendations: [
        "artifact_rec_1",
        "artifact_rec_2",
        "artifact_rec_3"
      ],
      action: "artifact_action",
    },
};
  
const DetailedReportView: FC<{ result: AnalysisResult }> = ({ result }) => {
    const { predicted_condition, confidence } = result;
    const details = conditionDetails[predicted_condition.toLowerCase()];
    const { t } = useTranslation();

    const getSeverity = (conf: number): string => {
        if (conf > 0.7) return t("confidence_high");
        if (conf > 0.3) return t("confidence_medium");
        return t("confidence_low");
    };
    const severity = getSeverity(confidence);

    if (!details) {
        return <div className="p-8 text-center bg-white dark:bg-slate-800 rounded-lg shadow-lg">{t('no_info_for_condition', { condition: predicted_condition })}</div>;
    }

  return (
        <div className="bg-white dark:bg-slate-800 p-6 sm:p-8 rounded-2xl shadow-lg border-t-8 border-brand-primary">
            <h2 className="text-2xl sm:text-3xl font-bold text-brand-primary mb-6">{t('detailed_report_title')}</h2>
            
            <div className="space-y-4 text-base sm:text-lg mb-8">
                <p><span className="font-semibold text-text-secondary dark:text-gray-400">{t('preliminary_diagnosis')}:</span> {t(details.name)}</p>
                <p><span className="font-semibold text-text-secondary dark:text-gray-400">{t('confidence')}:</span> {confidence.toFixed(2)} ({severity})</p>
            </div>
            
            <div className="space-y-8">
                <div>
                    <h3 className="text-xl sm:text-2xl font-semibold text-text-primary dark:text-white border-b-2 border-brand-primary/30 pb-2 mb-4">{t('explanation')}</h3>
                    <p className="text-text-secondary dark:text-gray-300 leading-relaxed text-sm sm:text-base">{t(details.description)}</p>
                </div>

                <div>
                    <h3 className="text-xl sm:text-2xl font-semibold text-text-primary dark:text-white border-b-2 border-brand-primary/30 pb-2 mb-4">{t('recommendations')}</h3>
                    <ul className="list-disc list-inside space-y-2 text-text-secondary dark:text-gray-300 text-sm sm:text-base">
                        {details.recommendations.map((rec, index) => <li key={index}>{t(rec)}</li>)}
                    </ul>
                </div>
            </div>

            <div className="mt-8 p-5 rounded-lg bg-gradient-to-r from-brand-gradient-start to-brand-gradient-end text-white shadow-md">
                <h4 className="font-bold text-lg sm:text-xl">{t('next_step')}</h4>
                <p className="mt-1 text-sm sm:text-base">{t(details.action)}</p>
            </div>

            <div className="mt-8 p-4 rounded-lg bg-danger-bg dark:bg-danger-bg/20 border border-danger-border text-danger-text dark:text-danger-border">
                <p className="font-semibold text-base sm:text-lg">{t('disclaimer_title')}</p>
                <p className="text-sm sm:text-base mt-1">{t('disclaimer_text')}</p>
            </div>
        </div>
    );
};

type UnderstandingTab = 'confidence' | 'types' | 'next_steps';

const UnderstandingResult: FC = () => {
    const [activeTab, setActiveTab] = useState<UnderstandingTab>('confidence');
    const { t } = useTranslation();

    return (
        <div className="mt-12">
            <div className="p-6 bg-page-bg dark:bg-slate-800/50 rounded-t-xl border-b-2 border-gray-200 dark:border-gray-700 flex items-center gap-4">
                <FileText className="h-6 w-6 sm:h-8 sm:w-8 text-brand-primary" />
                <h2 className="text-xl sm:text-2xl font-bold text-text-primary dark:text-white">{t('understanding_title')}</h2>
            </div>
            <div className="bg-white dark:bg-slate-800 p-6 rounded-b-xl shadow-lg">
                <div className="flex flex-wrap border-b border-slate-200 dark:border-slate-700 mb-6">
                    <UnderstandingTabButton icon={Gauge} title={t('tab_confidence')} isActive={activeTab === 'confidence'} onClick={() => setActiveTab('confidence')} />
                    <UnderstandingTabButton icon={ListTree} title={t('tab_conditions')} isActive={activeTab === 'types'} onClick={() => setActiveTab('types')} />
                    <UnderstandingTabButton icon={Wand} title={t('tab_next_steps')} isActive={activeTab === 'next_steps'} onClick={() => setActiveTab('next_steps')} />
                </div>
                <div>
                    {activeTab === 'confidence' && <ConfidenceView />}
                    {activeTab === 'types' && <ConditionTypesView />}
                    {activeTab === 'next_steps' && <NextStepsView />}
                </div>
            </div>
            <MedicalDisclaimer />
        </div>
    );
}

const UnderstandingTabButton: FC<{icon: ElementType, title: string, isActive: boolean, onClick: () => void}> = ({ icon: Icon, title, isActive, onClick }) => (
    <button onClick={onClick} className={`flex items-center gap-2 px-4 py-3 font-semibold text-base transition-colors duration-200 -mb-px ${ isActive ? 'border-b-2 border-brand-primary text-brand-primary' : 'text-text-secondary dark:text-gray-400 hover:text-brand-primary dark:hover:text-brand-primary'}`}>
        <Icon className="h-5 w-5" />
        <span>{title}</span>
    </button>
)

const ConfidenceView: FC = () => {
    const { t } = useTranslation();
    return (
        <div className="space-y-6">
            <ConfidenceItem 
                color="green" 
                title={t('confidence_high')}
                description={t('confidence_high_desc')}
            />
            <ConfidenceItem 
                color="yellow" 
                title={t('confidence_medium')}
                description={t('confidence_medium_desc')}
            />
            <ConfidenceItem 
                color="red" 
                title={t('confidence_low')}
                description={t('confidence_low_desc')}
            />
        </div>
    );
}

const ConfidenceItem: FC<{color: string, title: string, description: string}> = ({ color, title, description }) => {
    const colors = {
        green: { bg: 'bg-green-400', text: 'text-green-700', darkText: 'dark:text-green-300' },
        yellow: { bg: 'bg-yellow-400', text: 'text-yellow-700', darkText: 'dark:text-yellow-300' },
        red: { bg: 'bg-red-400', text: 'text-red-700', darkText: 'dark:text-red-300' },
    }
    const selectedColor = colors[color as keyof typeof colors] || colors.red;
    return (
        <div className="flex items-center gap-4">
            <div className={`w-5 h-5 rounded-full ${selectedColor.bg}`}></div>
            <div>
                <h4 className={`font-bold text-lg ${selectedColor.text} ${selectedColor.darkText}`}>{title}</h4>
                <p className="text-text-secondary dark:text-gray-400">{description}</p>
      </div>
    </div>
  );
};

const ConditionTypesView: FC = () => {
    const { t } = useTranslation();
    const types = [
        { icon: Heart, name: t("condition_normal"), description: t("condition_normal_desc"), color: 'text-normal', fill: 'fill-normal'},
        { icon: Heart, name: t("condition_murmur"), description: t("condition_murmur_desc"), color: 'text-murmur', fill: 'fill-murmur'},
        { icon: Heart, name: t("condition_extrastole"), description: t("condition_extrastole_desc"), color: 'text-extrasystole', fill: 'fill-extrasystole' },
        { icon: Heart, name: t("condition_extrahls"), description: t("condition_extrahls_desc"), color: 'text-extrahls', fill: 'fill-extrahls' },
        { icon: Volume2, name: t("condition_artifact"), description: t("condition_artifact_desc"), color: 'text-artifact', fill: 'fill-artifact' },
    ];
    return (
        <div className="space-y-4">
            {types.map(type => <ConditionTypeItem key={type.name} {...type} />)}
        </div>
    );
};

const ConditionTypeItem: FC<{icon: ElementType, name: string, description: string, color: string, fill: string}> = ({ icon: Icon, name, description, color, fill }) => (
    <div className="p-4 bg-page-bg/50 dark:bg-slate-700/50 rounded-lg flex items-center gap-4">
        <Icon className={`w-8 h-8 ${color} ${fill}`} strokeWidth={1.5}/>
        <div>
            <h4 className="font-bold text-lg text-text-primary dark:text-white">{name}</h4>
            <p className="text-text-secondary dark:text-gray-400">{description}</p>
      </div>
    </div>
);

const NextStepsView: FC = () => {
    const { t } = useTranslation();
    const steps = [
        { icon: FileText, text: t("next_steps_read_report") },
        { icon: Share2, text: t("next_steps_share") },
        { icon: Clipboard, text: t("next_steps_follow") },
        { icon: HeartPulse, text: t("next_steps_consult") },
    ];
    return (
        <div className="space-y-4">
             <div className="p-4 bg-danger-bg dark:bg-danger-bg/20 border-l-4 border-danger-border rounded-r-lg">
                <h3 className="text-xl font-bold text-danger-text dark:text-danger-border">{t('important_recommendation')}</h3>
            </div>
            {steps.map(step => <NextStepItem key={step.text} {...step} />)}
    </div>
  );
};

const NextStepItem: FC<{icon: ElementType, text: string}> = ({ icon: Icon, text }) => (
    <div className="p-4 bg-white dark:bg-slate-700/50 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 flex items-center gap-4">
        <Icon className="w-6 h-6 text-brand-primary" />
        <p className="font-semibold text-text-primary dark:text-gray-200">{text}</p>
    </div>
);

const MedicalDisclaimer: FC = () => {
    const { t } = useTranslation();
    return (
        <div className="mt-8 p-6 rounded-2xl bg-gradient-to-r from-brand-gradient-start to-brand-gradient-end text-white shadow-lg">
            <div className="flex items-start sm:items-center gap-3">
                <AlertTriangle className="w-10 h-10 sm:w-8 sm:h-8 flex-shrink-0" />
                <h3 className="text-lg sm:text-xl font-bold">{t('medical_disclaimer_title')}</h3>
            </div>
            <p className="mt-3 leading-relaxed text-sm sm:text-base">
                {t('medical_disclaimer_text')}
            </p>
        </div>
    );
}


// --- Assistant Panel (NEW DESIGN) ---
const AssistantPanel: FC = () => {
    const { t } = useTranslation();
    const initialMessage: ChatMessage = { 
      role: 'bot', 
      text: t('initial_bot_message')
    };

    const [messages, setMessages] = useState<ChatMessage[]>([initialMessage]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
    const chatContainerRef = useRef<HTMLDivElement>(null);
  
    useEffect(() => { 
        chatContainerRef.current?.scrollTo({ top: chatContainerRef.current.scrollHeight, behavior: 'smooth' }); 
    }, [messages]);
  
    const handleSubmit = async (userMessage: string) => {
        if (!userMessage.trim() || isLoading) return;
        
        const newMessages: ChatMessage[] = [...messages, { role: 'user', text: userMessage }];
        setMessages(newMessages);
    setInput('');
    setIsLoading(true);

        try {
            const response = await callBackendWithRetry<{response: string}>('/api/assistant', {
                method: 'POST',
                data: { 
                    message: userMessage,
                    history: messages.map(m => ({ role: m.role, text: m.text })) // Sending history
                }
            });
            setMessages(prev => [...prev, { role: 'bot', text: response.data.response }]);
    } catch (error) {
            setMessages(prev => [...prev, { role: 'bot', text: t('server_error') }]); 
    } finally {
        setIsLoading(false);
        }
    };

    const handleQuickQuestion = (question: string) => {
        handleSubmit(question);
    };
    
    const handleClearChat = () => {
        setMessages([initialMessage]);
    };

    return (
        <div className="space-y-8 animate-fade-in">
            <AssistantHeader />
            <QuickQuestionsV2 onQuestionClick={handleQuickQuestion} />
            <ChatWrapper 
                messages={messages}
                input={input}
                setInput={setInput}
                isLoading={isLoading}
                onSubmit={handleSubmit}
                chatContainerRef={chatContainerRef}
            />
            <ConversationTips onClearChat={handleClearChat} />
        </div>
    );
};

const AssistantHeader: FC = () => {
    const { t } = useTranslation();
    return (
        <div className="p-8 sm:p-10 rounded-2xl bg-gradient-to-br from-brand-gradient-start to-brand-gradient-end text-white text-center shadow-xl">
            <h2 className="text-3xl sm:text-4xl font-bold">{t('assistant_header')}</h2>
            <p className="mt-2 text-base sm:text-lg text-purple-200">{t('assistant_tagline')}</p>
        </div>
    );
}

const QuickQuestionsV2: FC<{ onQuestionClick: (q: string) => void }> = ({ onQuestionClick }) => {
    const { t } = useTranslation();
    const questions = [
        { icon: Salad, text: t("q_food") },
        { icon: Dumbbell, text: t("q_exercise") },
        { icon: Stethoscope, text: t("q_murmur") },
        { icon: Siren, text: t("q_symptoms") },
        { icon: BrainCircuit, text: t("q_stress") },
        { icon: UserCheck, text: t("q_self_check") },
    ];
  return (
        <div className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm border border-gray-200 dark:border-slate-700 p-6 rounded-2xl shadow-lg">
            <h3 className="text-lg sm:text-xl font-bold text-text-primary dark:text-white flex items-center gap-2 mb-4">
                <MessageCircle className="text-brand-primary" />
                {t('faq')}
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {questions.map(({ icon: Icon, text }) => (
                    <button 
                        key={text} 
                        onClick={() => onQuestionClick(text)}
                        className="p-4 bg-gradient-to-r from-brand-gradient-start to-brand-gradient-end text-white rounded-lg flex items-center gap-3 text-left font-semibold hover:scale-105 hover:shadow-xl transition-all duration-300 text-sm sm:text-base"
                    >
                        <Icon className="w-6 h-6 flex-shrink-0" />
                        <span>{text}</span>
                    </button>
                ))}
      </div>
        </div>
    );
};

const ChatWrapper: FC<{
    messages: ChatMessage[],
    input: string,
    setInput: Dispatch<SetStateAction<string>>,
    isLoading: boolean,
    onSubmit: (msg: string) => void,
    chatContainerRef: React.RefObject<HTMLDivElement>
}> = ({ messages, input, setInput, isLoading, onSubmit, chatContainerRef }) => {
    const { t } = useTranslation();
    return (
        <div className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm border border-gray-200 dark:border-slate-700 p-6 rounded-2xl shadow-lg">
            <h3 className="text-lg sm:text-xl font-bold text-text-primary dark:text-white flex items-center gap-2 mb-4">
                <MessageCircle className="text-brand-primary" />
                {t('chat_room')}
            </h3>
            <div ref={chatContainerRef} className="h-[300px] sm:h-[400px] overflow-y-auto pr-4 space-y-4">
                {messages.map((msg, i) => <MessageBubbleV2 key={i} msg={msg} />)}
                {isLoading && <TypingIndicatorV2 />}
      </div>
            <form onSubmit={(e) => { e.preventDefault(); onSubmit(input); }} className="mt-4 flex items-center gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
                    placeholder={t('type_your_question')}
                    className="flex-grow p-3 bg-page-bg dark:bg-slate-700 border border-gray-200 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-brand-primary/50 transition-all dark:text-white" 
            disabled={isLoading}
          />
                <button type="submit" disabled={isLoading} className="bg-gradient-to-r from-brand-gradient-start to-brand-gradient-end text-white rounded-lg p-3 hover:scale-110 disabled:opacity-50 disabled:scale-100 transition-all duration-200 flex items-center justify-center w-12 h-12">
                    {isLoading ? <Loader className="animate-spin" size={24} /> : <Sparkles size={24} />}
          </button>
        </form>
      </div>
    );
}

const MessageBubbleV2: FC<{ msg: ChatMessage }> = ({ msg }) => {
    const isBot = msg.role === 'bot';
    const { t } = useTranslation();

    if (isBot) {
        return (
            <div className="flex justify-start w-full">
                <div className="flex flex-col items-start gap-2">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-brand-gradient-start to-brand-gradient-end flex items-center justify-center text-white flex-shrink-0 self-start">
                        <Bot size={24} />
                    </div>
                    <div className="max-w-xl p-4 rounded-2xl bg-page-bg dark:bg-slate-700 text-text-primary dark:text-gray-200 rounded-tl-none">
                        <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                        <p className="text-xs text-text-secondary dark:text-gray-400 mt-2">{t('powered_by')}</p>
                    </div>
                </div>
            </div>
        );
    }

    // User message
    return (
        <div className="flex justify-end w-full">
            <div className="flex flex-col items-end gap-2">
                <div className="w-10 h-10 rounded-full bg-gray-300 dark:bg-slate-600 flex items-center justify-center text-text-secondary dark:text-gray-300 flex-shrink-0">
                    <User size={24} />
                </div>
                <div className="max-w-xl p-4 rounded-2xl bg-brand-primary text-white rounded-br-none">
                    <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                </div>
            </div>
        </div>
    );
};

const TypingIndicatorV2: FC = () => (
    <div className="flex justify-start w-full">
        <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-end sm:gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-brand-gradient-start to-brand-gradient-end flex items-center justify-center text-white flex-shrink-0"><Bot size={24} /></div>
            <div className="p-4 rounded-2xl bg-page-bg dark:bg-slate-700 rounded-bl-none flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                <span className="h-2.5 w-2.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                <span className="h-2.5 w-2.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"></span>
      </div>
        </div>
    </div>
);

const ConversationTips: FC<{onClearChat: () => void}> = ({ onClearChat }) => {
    const { t } = useTranslation();
    return (
        <div className="space-y-6">
            <div className="p-6 rounded-2xl bg-gradient-to-br from-brand-gradient-start to-brand-gradient-end text-white shadow-xl">
                 <h3 className="text-xl font-bold flex items-center gap-2 mb-4">
                    <Lightbulb />
                    {t('conversation_tips')}
                </h3>
                <div className="space-y-3">
                    <TipItem icon={Crosshair} title={t('tip_specific')} description={t('tip_specific_desc')} />
                    <TipItem icon={Gem} title={t('tip_more_info')} description={t('tip_more_info_desc')} />
                    <TipItem icon={ShieldCheck} title={t('tip_info_only')} description={t('tip_info_only_desc')} />
                </div>
            </div>
            <button onClick={onClearChat} className="w-full p-4 bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm border border-gray-200 dark:border-slate-700 rounded-lg shadow-md font-semibold text-danger-text flex items-center justify-center gap-2 hover:bg-danger-bg dark:hover:bg-danger-bg/20 transition-colors">
                <Trash2 size={20} />
                {t('clear_chat')}
            </button>
        </div>
    );
}

const TipItem: FC<{icon: ElementType, title: string, description: string}> = ({icon: Icon, title, description}) => (
    <div className="p-3 bg-white/20 rounded-lg flex items-center gap-3">
        <Icon className="w-6 h-6 flex-shrink-0" />
        <div>
            <h4 className="font-bold">{title}</h4>
            <p className="text-sm text-purple-200">{description}</p>
        </div>
    </div>
)

const ErrorMessage: FC<{ message: string }> = ({ message }) => (
    <div className="mt-4 flex items-center gap-2 text-danger-text bg-danger-bg dark:bg-danger-bg/20 border border-danger-border rounded-md p-3">
        <AlertCircle size={20} />
        <p className="text-sm font-semibold">{message}</p>
    </div>
);

export default App; 