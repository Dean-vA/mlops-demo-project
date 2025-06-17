// Audio Input Component - Handles file upload and microphone recording
class AudioInputComponent {
    constructor(state, services) {
        this.state = state;
        this.services = services;
        this.container = document.getElementById('audio-input-component');
        this.currentTab = 'file';

        this.elements = {};
        this.isRecording = false;
    }

    async init() {
        this.render();
        this.setupEventListeners();
        this.setupAudioServiceListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="recording-section">
                <div class="recording-tabs">
                    <button class="tab-button active" data-tab="file">📁 Audio File</button>
                    <button class="tab-button" data-tab="mic">🎤 Microphone</button>
                </div>

                <!-- File Upload Tab -->
                <div class="tab-content active" data-tab-content="file">
                    <div class="upload-area" id="upload-area">
                        <div class="upload-icon">📁</div>
                        <div class="upload-text">Drop your audio file here or click to browse</div>
                        <div class="upload-subtext">Supports .wav and .flac files (max ${UIUtils.formatFileSize(AppConfig.UI.MAX_FILE_SIZE)})</div>
                        <input type="file" id="file-input" accept=".wav,.flac" style="display: none;">
                    </div>

                    <div class="file-info" id="file-info" style="display: none;">
                        <strong>Selected file:</strong> <span id="file-name"></span><br>
                        <strong>Size:</strong> <span id="file-size"></span><br>
                        <strong>Type:</strong> <span id="file-type"></span>
                        <button class="replay-btn" id="play-file-btn" style="margin-top: 10px;">🔊 Play File</button>
                    </div>
                </div>

                <!-- Microphone Tab -->
                <div class="tab-content" data-tab-content="mic">
                    <div class="mic-area" id="mic-area">
                        <div class="mic-icon" id="mic-icon">🎤</div>
                        <div class="mic-text" id="mic-text">Click to start recording</div>
                        <div class="mic-subtext">Record directly from your microphone</div>

                        <div class="recording-controls" id="recording-controls" style="display: none;">
                            <div class="recording-timer" id="recording-timer">00:00</div>
                            <div class="recording-visualizer">
                                <div class="wave-bar"></div>
                                <div class="wave-bar"></div>
                                <div class="wave-bar"></div>
                                <div class="wave-bar"></div>
                                <div class="wave-bar"></div>
                            </div>
                        </div>
                    </div>

                    <div class="recorded-info" id="recorded-info" style="display: none;">
                        <strong>Recording:</strong> <span id="recorded-duration"></span><br>
                        <strong>Quality:</strong> 44.1kHz, 16-bit WAV<br>
                        <button class="replay-btn" id="replay-btn">🔊 Play Recording</button>
                        <button class="rerecord-btn" id="rerecord-btn">🔄 Record Again</button>
                    </div>
                </div>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            // Tab elements
            tabButtons: this.container.querySelectorAll('.tab-button'),
            tabContents: this.container.querySelectorAll('.tab-content'),

            // File upload elements
            uploadArea: this.container.querySelector('#upload-area'),
            fileInput: this.container.querySelector('#file-input'),
            fileInfo: this.container.querySelector('#file-info'),
            fileName: this.container.querySelector('#file-name'),
            fileSize: this.container.querySelector('#file-size'),
            fileType: this.container.querySelector('#file-type'),
            playFileBtn: this.container.querySelector('#play-file-btn'),

            // Microphone elements
            micArea: this.container.querySelector('#mic-area'),
            micIcon: this.container.querySelector('#mic-icon'),
            micText: this.container.querySelector('#mic-text'),
            recordingControls: this.container.querySelector('#recording-controls'),
            recordingTimer: this.container.querySelector('#recording-timer'),
            recordedInfo: this.container.querySelector('#recorded-info'),
            recordedDuration: this.container.querySelector('#recorded-duration'),
            replayBtn: this.container.querySelector('#replay-btn'),
            rerecordBtn: this.container.querySelector('#rerecord-btn')
        };
    }

    setupEventListeners() {
        // Tab switching
        this.elements.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                this.switchTab(button.dataset.tab);
            });
        });

        // File upload
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });

        this.elements.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.elements.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.elements.uploadArea.addEventListener('drop', this.handleDrop.bind(this));

        this.elements.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.elements.playFileBtn.addEventListener('click', this.playSelectedFile.bind(this));

        // Microphone
        this.elements.micArea.addEventListener('click', this.handleMicClick.bind(this));
        this.elements.replayBtn.addEventListener('click', this.playRecordedAudio.bind(this));
        this.elements.rerecordBtn.addEventListener('click', this.startNewRecording.bind(this));
    }

    setupAudioServiceListeners() {
        // Recording events
        document.addEventListener('recordingStarted', () => {
            this.updateRecordingUI(true);
        });

        document.addEventListener('recordingStopped', () => {
            this.updateRecordingUI(false);
        });

        document.addEventListener('recordingComplete', (event) => {
            this.handleRecordingComplete(event.detail.audioBlob);
        });

        document.addEventListener('recordingTimerUpdate', (event) => {
            this.elements.recordingTimer.textContent = event.detail.timeString;
        });
    }

    // Tab switching
    switchTab(tabName) {
        this.currentTab = tabName;

        // Update tab buttons
        this.elements.tabButtons.forEach(button => {
            button.classList.toggle('active', button.dataset.tab === tabName);
        });

        // Update tab contents
        this.elements.tabContents.forEach(content => {
            content.classList.toggle('active', content.dataset.tabContent === tabName);
        });

        // Reset state when switching tabs
        this.state.reset();
    }

    // File upload handlers
    handleDragOver(e) {
        e.preventDefault();
        this.elements.uploadArea.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.elements.uploadArea.classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        this.elements.uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFile(file) {
        const validationError = this.services.audio.getFileValidationError(file);
        if (validationError) {
            this.showError(validationError);
            return;
        }

        this.state.setSelectedFile(file);
        this.updateFileUI(file);
    }

    updateFileUI(file) {
        this.elements.fileName.textContent = file.name;
        this.elements.fileSize.textContent = UIUtils.formatFileSize(file.size);
        this.elements.fileType.textContent = file.type || 'audio/' + file.name.split('.').pop();

        this.elements.uploadArea.classList.add('has-file');
        this.elements.uploadArea.querySelector('.upload-icon').textContent = '🎵';
        this.elements.uploadArea.querySelector('.upload-text').textContent = 'Audio file selected';

        this.elements.fileInfo.style.display = 'block';
    }

    async playSelectedFile() {
        if (!this.state.selectedFile) return;

        try {
            await this.services.audio.playAudioFile(this.state.selectedFile);
            UIUtils.showToast('Playing audio file...', 'info', 2000);
        } catch (error) {
            console.error('Error playing file:', error);
            this.showError('Could not play audio file');
        }
    }

    // Microphone handlers
    async handleMicClick() {
        if (this.services.audio.isRecording()) {
            this.services.audio.stopRecording();
        } else {
            try {
                await this.services.audio.startRecording();
            } catch (error) {
                this.showError(error.message);
            }
        }
    }

    updateRecordingUI(isRecording) {
        this.isRecording = isRecording;

        if (isRecording) {
            this.elements.micArea.classList.add('recording');
            this.elements.micIcon.textContent = '⏹️';
            this.elements.micText.textContent = 'Click to stop recording';
            this.elements.recordingControls.style.display = 'flex';
        } else {
            this.elements.micArea.classList.remove('recording');
            this.elements.recordingControls.style.display = 'none';
        }
    }

    handleRecordingComplete(audioBlob) {
        this.state.setRecordedAudio(audioBlob);

        const duration = this.elements.recordingTimer.textContent;
        this.elements.recordedDuration.textContent = duration;
        this.elements.recordedInfo.style.display = 'block';

        this.elements.micIcon.textContent = '✅';
        this.elements.micText.textContent = 'Recording complete';

        UIUtils.showToast('Recording completed successfully!', 'success');
    }

    async playRecordedAudio() {
        if (!this.state.recordedAudioBlob) return;

        try {
            await this.services.audio.playAudioFile(this.state.recordedAudioBlob);
            UIUtils.showToast('Playing recorded audio...', 'info', 2000);
        } catch (error) {
            console.error('Error playing recording:', error);
            this.showError('Could not play recorded audio');
        }
    }

    startNewRecording() {
        this.state.setRecordedAudio(null);
        this.elements.recordedInfo.style.display = 'none';
        this.elements.micIcon.textContent = '🎤';
        this.elements.micText.textContent = 'Click to start recording';
        this.state.reset();
    }

    // Reset file upload UI
    resetFileUI() {
        this.elements.uploadArea.classList.remove('has-file');
        this.elements.uploadArea.querySelector('.upload-icon').textContent = '📁';
        this.elements.uploadArea.querySelector('.upload-text').textContent = 'Drop your audio file here or click to browse';
        this.elements.fileInfo.style.display = 'none';
        this.elements.fileInput.value = '';
    }

    // Reset recording UI
    resetRecordingUI() {
        this.elements.recordedInfo.style.display = 'none';
        this.elements.micIcon.textContent = '🎤';
        this.elements.micText.textContent = 'Click to start recording';
        this.elements.recordingControls.style.display = 'none';
        this.elements.micArea.classList.remove('recording');
    }

    // Public methods
    reset() {
        this.resetFileUI();
        this.resetRecordingUI();
        this.services.audio.cleanup();
    }

    showError(message) {
        const event = new CustomEvent('showError', {
            detail: { message }
        });
        document.dispatchEvent(event);
    }

    // Get current audio input
    getCurrentAudioInput() {
        return this.state.getAudioInput();
    }

    // Check if audio input is available
    hasAudioInput() {
        return this.state.hasAudioInput();
    }
}
