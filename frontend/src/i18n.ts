import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import enTranslation from './locales/en/translation.json';
import thTranslation from './locales/th/translation.json';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    debug: true,
    fallbackLng: 'th',
    interpolation: {
      escapeValue: false,
    },
    resources: {
      en: {
        translation: enTranslation.translation,
      },
      th: {
        translation: thTranslation.translation,
      },
    },
  });

export default i18n; 