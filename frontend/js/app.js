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
        this.components = {}; // Add components reference

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
        this.initializationComplete = false;
    }

    async init() {
        try {
            console.log('🚀 Initializing Parakeet STT Application...');

            // Check if required elements exist
            const requiredElements = [
                'audio-input-component',
                'settings-component',
                'action-buttons-component',
                'download-component',
                'loading-component',
                'results-component',
                'error-component',
                'status-indicator-component',
                'audio-notification-component'
            ];

            const missingElements = requiredElements.filter(id => !document.getElementById(id));

            if (missingElements.length > 0) {
                throw new Error(`Missing required DOM elements: ${missingElements.join(', ')}`);
            }

            // Initialize services first
            console.log('📡 Initializing services...');
            this.services.api = new ApiService();
            this.services.audio = new AudioService();

            // Initialize components with error handling
            console.log('🎨 Initializing components...');
            await this.initializeComponents();

            // Set up state listeners
            this.setupStateListeners();

            // Store components reference in state for cross-component communication
            this.state.components = this.components;

            // Start health monitoring
            console.log('🏥 Starting API health monitoring...');
            this.services.api.startHealthMonitoring();

            this.initializationComplete = true;
            console.log('✅ Parakeet STT Application initialized successfully');

        } catch (error) {
            console.error('❌ Failed to initialize application:', error);

            // Show error in a simple way if components aren't ready
            const errorDiv = document.createElement('div');
            errorDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #dc3545;
                max-width: 400px;
                z-index: 10000;
            `;
            errorDiv.innerHTML = `
                <strong>Initialization Error:</strong><br>
                ${error.message}<br>
                <small>Check browser console for details.</small>
            `;
            document.body.appendChild(errorDiv);

            // Try to show error through component if available
            if (this.components.error) {
                this.components.error.show('Failed to initialize application: ' + error.message);
            }
        }
    }

    async initializeComponents() {
        const componentConfigs = [
            { name: 'audioInput', class: AudioInputComponent, args: [this.state, this.services] },
            { name: 'settings', class: SettingsComponent, args: [this.state] },
            { name: 'actionButtons', class: ActionButtonsComponent, args: [this.state, this.services] },
            { name: 'download', class: DownloadComponent, args: [this.state] },
            { name: 'loading', class: LoadingComponent, args: [] },
            { name: 'results', class: ResultsComponent, args: [this.state, this.services] },
            { name: 'error', class: ErrorComponent, args: [] },
            { name: 'statusIndicator', class: StatusIndicatorComponent, args: [] },
            { name: 'audioNotification', class: AudioNotificationComponent, args: [] }
        ];

        // Initialize components one by one with error handling
        for (const config of componentConfigs) {
            try {
                console.log(`  - Initializing ${config.name}...`);

                // Check if class exists
                if (typeof config.class !== 'function') {
                    console.error(`⚠️ Component class ${config.class.name || 'unknown'} not found`);
                    continue;
                }

                // Create component instance
                this.components[config.name] = new config.class(...config.args);

                // Initialize if init method exists
                if (typeof this.components[config.name].init === 'function') {
                    await this.components[config.name].init();
                }

                console.log(`  ✅ ${config.name} initialized`);

            } catch (error) {
                console.error(`  ❌ Failed to initialize ${config.name}:`, error);
                // Continue with other components rather than failing completely
                continue;
            }
        }

        console.log('🎨 Component initialization completed');
    }

    setupStateListeners() {
        // Listen to state changes and update UI accordingly
        this.state.on('fileSelected', (file) => {
            if (this.components.actionButtons) {
                this.components.actionButtons.enable();
            }
        });

        this.state.on('audioRecorded', (blob) => {
            if (this.components.actionButtons) {
                this.components.actionButtons.enable();
            }
        });

        this.state.on('processingStateChanged', (isProcessing) => {
            if (isProcessing) {
                if (this.components.loading) this.components.loading.show();
                if (this.components.actionButtons) this.components.actionButtons.disable();
                if (this.components.results) this.components.results.hide();
                if (this.components.download) this.components.download.hide();
            } else {
                if (this.components.loading) this.components.loading.hide();
                if (this.components.actionButtons) this.components.actionButtons.enable();
            }
        });

        this.state.on('transcriptionComplete', (data) => {
            if (this.components.results) {
                this.components.results.display(data);
            }
            if (this.components.download) {
                this.components.download.show();
            }
        });

        this.state.on('stateReset', () => {
            if (this.components.results) this.components.results.hide();
            if (this.components.download) this.components.download.hide();
            if (this.components.error) this.components.error.hide();
            if (this.components.loading) this.components.loading.hide();

            if (!this.state.hasAudioInput() && this.components.actionButtons) {
                this.components.actionButtons.disable();
            }
        });
    }

    // Public methods for component communication
    async transcribe(mode = 'transcribe') {
        if (!this.initializationComplete) {
            console.warn('Application not fully initialized yet');
            return;
        }

        if (!this.state.hasAudioInput() || this.state.isProcessing) {
            return;
        }

        try {
            this.state.setProcessing(true);
            this.state.currentMode = mode;

            const audioInput = this.state.getAudioInput();
            const settings = this.components.settings ? this.components.settings.getSettings() : {};

            let result;
            if (mode === 'diarize') {
                result = await this.services.api.transcribeAndDiarize(audioInput, settings);
            } else {
                result = await this.services.api.transcribe(audioInput, settings);
            }

            this.state.setTranscriptionData(result);
            if (this.components.error) {
                this.components.error.hide();
            }

        } catch (error) {
            console.error('Transcription error:', error);
            if (this.components.error) {
                this.components.error.show(this.formatError(error));
            }
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
    // Add a small delay to ensure all resources are loaded
    setTimeout(() => {
        window.app = new App();
        window.app.init();
    }, 100);
});

// Handle unhandled errors
window.addEventListener('error', (event) => {
    console.error('Unhandled error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});
