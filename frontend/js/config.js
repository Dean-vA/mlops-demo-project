// Application Configuration
window.AppConfig = {
    API: {
        BASE_URL: (() => {
            const hostname = window.location.hostname;

            // Local development
            if (hostname === 'localhost' || hostname === '127.0.0.1') {
                return 'http://localhost:3569';
            }

            // Local network IPs (226, 227, 228, etc.)
            if (hostname.startsWith('194.171.191.')) {
                return `http://${hostname}:3569`;
            }

            // Azure Container Apps - detect backend URL from frontend URL
            if (hostname.includes('azurecontainerapps.io')) {
                // Replace 'frontend' with 'backend' in the hostname
                const backendHostname = hostname.replace('frontend', 'backend');
                return `https://${backendHostname}`;
            }

            // Default fallback
            return 'http://194.171.191.227:3569';
        })(),
        ENDPOINTS: {
            HEALTH: '/health',
            GPU_INFO: '/gpu-info',
            TRANSCRIBE: '/transcribe',
            DIARIZE: '/diarize',
            TRANSCRIBE_AND_DIARIZE: '/transcribe_and_diarize',
            SUMMARIZE: '/summarize'
        },
        TIMEOUT: 600000 // 10 minutes
    },

    AUDIO: {
        SUPPORTED_FORMATS: ['.wav', '.flac'],
        MIME_TYPES: ['audio/wav', 'audio/flac', 'audio/webm'],
        RECORDING: {
            SAMPLE_RATE: 44100,
            CHANNEL_COUNT: 1,
            ECHO_CANCELLATION: true,
            NOISE_SUPPRESSION: true
        }
    },

    UI: {
        HEALTH_CHECK_INTERVAL: 30000, // 30 seconds
        NOTIFICATION_DURATION: 3000, // 3 seconds
        BATCH_SIZE: 50, // For processing large datasets
        MAX_FILE_SIZE: 500 * 1024 * 1024 // 500MB
    },

    FEATURES: {
        SPEAKER_DIARIZATION: true,
        WORD_TIMESTAMPS: true,
        SEGMENT_PLAYBACK: true,
        BATCH_PROCESSING: false // Future feature
    },

    SPEAKER_COLORS: [
        'speaker-0', 'speaker-1', 'speaker-2', 'speaker-3', 'speaker-4',
        'speaker-5', 'speaker-6', 'speaker-7', 'speaker-8', 'speaker-9', 'speaker-10'
    ]
};
