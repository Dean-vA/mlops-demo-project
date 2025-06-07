// Application State Manager and Main Controller
class AppState {
    constructor() {
        this.selectedFile = null;
        this.recordedAudioBlob = null;
        this.currentTranscriptionData = null;
        this.speakerNames = {};
        this.currentMode = 'transcribe';
        this.isProcessing = false;
        this.currentAudio = null;
        this.currentlyPlayingSegment = null;
        this.currentlyEditingLabel = null;

        // Event emitter for state changes
        this.listeners = {};
    }

    // Event system
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    // State setters with event emission
    setSelectedFile(file) {
        this.selectedFile = file;
        this.emit('fileSelected', file);
    }

    setRecordedAudio(blob) {
        this.recordedAudioBlob = blob;
        this.emit('audioRecorded', blob);
    }

    setTranscriptionData(data) {
        this.currentTranscriptionData = data;
        this.emit('transcriptionComplete', data);
    }

    setProcessing(isProcessing) {
        this.isProcessing = isProcessing;
        this.emit('processingStateChanged', isProcessing);
    }

    setSpeakerNames(names) {
        this.speakerNames = { ...names };
        this.emit('speakerNamesChanged', this.speakerNames);
    }

    reset() {
        this.selectedFile = null;
        this.recordedAudioBlob = null;
        this.currentTranscriptionData = null;
        this.speakerNames = {};
        this.currentMode = 'transcribe';
        this.isProcessing = false;
        this.currentAudio = null;
        this.currentlyPlayingSegment = null;
        this.currentlyEditingLabel = null;
        this.emit('stateReset');
    }

    // Getters
    hasAudioInput() {
        return this.selectedFile || this.recordedAudioBlob;
    }

    getAudioInput() {
        return this.recordedAudioBlob || this.selectedFile;
    }

    getDisplayName(speakerId) {
        return this.speakerNames[speakerId] || speakerId;
    }
}

// Main Application Controller
class App {
    constructor() {
        this.state = new AppState();
        this.components = {};
        this.services = {};

        this.init();
    }

    async init() {
        try {
            // Initialize services
            this.services.api = new ApiService();
            this.services.audio = new AudioService();

            // Initialize components
            await this.initializeComponents();

            // Set up state listeners
            this.setupStateListeners();

            // Start health monitoring
            this.services.api.startHealthMonitoring();

            console.log('🚀 Parakeet STT Application initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize application:', error);
            this.components.error?.show('Failed to initialize application: ' + error.message);
        }
    }


    async initializeComponents() {
        // Initialize all components
        this.components.audioInput = new AudioInputComponent(this.state, this.services);
        this.components.settings = new SettingsComponent(this.state);
        this.components.actionButtons = new ActionButtonsComponent(this.state, this.services);
        this.components.download = new DownloadComponent(this.state);
        this.components.loading = new LoadingComponent();
        this.components.results = new ResultsComponent(this.state, this.services);
        this.components.summary = new SummaryComponent(this.state, this.services);
        this.components.error = new ErrorComponent();
        this.components.statusIndicator = new StatusIndicatorComponent();
        this.components.audioNotification = new AudioNotificationComponent();

        // Initialize all components
        await Promise.all(
            Object.values(this.components).map(component =>
                component.init ? component.init() : Promise.resolve()
            )
        );
    }

    setupStateListeners() {
        // Listen to state changes and update UI accordingly
        this.state.on('fileSelected', (file) => {
            this.components.actionButtons.enable();
        });

        this.state.on('audioRecorded', (blob) => {
            this.components.actionButtons.enable();
        });

        this.state.on('processingStateChanged', (isProcessing) => {
            if (isProcessing) {
                this.components.loading.show();
                this.components.actionButtons.disable();
                this.components.results.hide();
                this.components.download.hide();
            } else {
                this.components.loading.hide();
                this.components.actionButtons.enable();
            }
        });

        this.state.on('transcriptionComplete', (data) => {
            this.components.results.display(data);
            this.components.download.show();
        });

        this.state.on('stateReset', () => {
            this.components.results.hide();
            this.components.download.hide();
            this.components.error.hide();
            this.components.loading.hide();

            if (!this.state.hasAudioInput()) {
                this.components.actionButtons.disable();
            }
        });
    }

    // Public methods for component communication
    async transcribe(mode = 'transcribe') {
        if (!this.state.hasAudioInput() || this.state.isProcessing) {
            return;
        }

        try {
            this.state.setProcessing(true);
            this.state.currentMode = mode;

            const audioInput = this.state.getAudioInput();
            const settings = this.components.settings.getSettings();

            let result;
            if (mode === 'diarize') {
                result = await this.services.api.transcribeAndDiarize(audioInput, settings);
            } else {
                result = await this.services.api.transcribe(audioInput, settings);
            }

            this.state.setTranscriptionData(result);
            this.components.error.hide();

        } catch (error) {
            console.error('Transcription error:', error);
            this.components.error.show(this.formatError(error));
        } finally {
            this.state.setProcessing(false);
        }
    }

    formatError(error) {
        if (error.response) {
            return `API Error: ${error.response.data.detail || error.response.statusText}`;
        } else if (error.request) {
            return 'Network error: Could not connect to the API. Make sure the backend is running.';
        } else {
            return `Error: ${error.message}`;
        }
    }

    reset() {
        this.state.reset();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
