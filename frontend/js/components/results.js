// Enhanced Results Component with Collapsible Sections
class ResultsComponent {
    constructor(state, services) {
        this.state = state;
        this.services = services;
        this.container = document.getElementById('results-component');
        this.currentlyEditingLabel = null;
        this.currentData = null;
        this.collapsedSections = new Set(); // Track collapsed sections
    }

    async init() {
        this.render();
        this.setupEventListeners();
        // Don't load collapsed state here - will be loaded when showing results
    }

    render() {
        this.container.innerHTML = `
            <div class="results" id="results" style="display: none;">
                <!-- Results Tabs -->
                <div class="results-tabs" id="results-tabs">
                    <button class="tab-button active" data-tab="transcription">📝 Transcription</button>
                    <button class="tab-button" data-tab="analysis" id="analysis-tab" style="display: none;">📊 Analysis</button>
                    <button class="tab-button" data-tab="summary" id="summary-tab" style="display: none;">🐉 Summary</button>
                </div>

                <!-- Transcription Tab -->
                <div class="tab-content active" data-tab-content="transcription">
                    <!-- Main Transcription -->
                    <div class="result-section collapsible-section" data-section="transcription">
                        <h3 class="collapsible-header" data-target="transcription">
                            <span class="collapse-indicator">▼</span>
                            📝 Transcription
                        </h3>
                        <div class="collapsible-content" data-content="transcription">
                            <div class="transcription-text" id="transcription-text"></div>
                            <button class="btn btn-secondary" id="copy-transcription-btn" style="margin-top: 10px;">
                                📋 Copy Text
                            </button>
                        </div>
                    </div>

                    <!-- Statistics -->
                    <div class="stats" id="stats">
                        <div class="stat-item">
                            <div class="stat-value" id="processing-time">-</div>
                            <div class="stat-label">Processing Time (s)</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="word-count">-</div>
                            <div class="stat-label">Words</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="segment-count">-</div>
                            <div class="stat-label">Segments</div>
                        </div>
                        <div class="stat-item" id="speaker-count-stat" style="display: none;">
                            <div class="stat-value" id="speaker-count">-</div>
                            <div class="stat-label">Speakers</div>
                        </div>
                    </div>

                    <!-- Segments -->
                    <div class="result-section" id="segments-section" style="display: none;">
                        <h3>📑 Segments <span class="subtitle">Click to play audio</span></h3>
                        <div class="segments" id="segments"></div>
                    </div>

                    <!-- Word Timestamps -->
                    <div class="result-section" id="timestamps-section" style="display: none;">
                        <h3>⏱️ Word Timestamps</h3>
                        <div class="timestamps" id="timestamps"></div>
                    </div>
                </div>

                <!-- Analysis Tab -->
                <div class="tab-content" data-tab-content="analysis">
                    <!-- Speaker Summary -->
                    <div class="result-section collapsible-section" id="speaker-summary-section" data-section="speaker-summary" style="display: none;">
                        <h3 class="collapsible-header" data-target="speaker-summary">
                            <span class="collapse-indicator">▼</span>
                            👥 Speaker Summary
                            <span class="subtitle">Click names to edit</span>
                        </h3>
                        <div class="collapsible-content" data-content="speaker-summary">
                            <div class="speaker-summary" id="speaker-summary"></div>
                        </div>
                    </div>
                </div>

                <!-- Summary Tab -->
                <div class="tab-content" data-tab-content="summary">
                    <!-- Summary component will be injected here -->
                    <div id="summary-component"></div>
                </div>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            results: this.container.querySelector('#results'),
            transcriptionText: this.container.querySelector('#transcription-text'),
            copyTranscriptionBtn: this.container.querySelector('#copy-transcription-btn'),

            // Stats
            processingTime: this.container.querySelector('#processing-time'),
            wordCount: this.container.querySelector('#word-count'),
            segmentCount: this.container.querySelector('#segment-count'),
            speakerCount: this.container.querySelector('#speaker-count'),
            speakerCountStat: this.container.querySelector('#speaker-count-stat'),

            // Sections
            speakerSummarySection: this.container.querySelector('#speaker-summary-section'),
            speakerSummary: this.container.querySelector('#speaker-summary'),
            segmentsSection: this.container.querySelector('#segments-section'),
            segments: this.container.querySelector('#segments'),
            timestampsSection: this.container.querySelector('#timestamps-section'),
            timestamps: this.container.querySelector('#timestamps'),

            // Collapsible elements
            collapsibleHeaders: this.container.querySelectorAll('.collapsible-header'),
            collapsibleSections: this.container.querySelectorAll('.collapsible-section')
        };
    }

    setupEventListeners() {
        // Copy transcription button
        this.elements.copyTranscriptionBtn.addEventListener('click', () => {
            this.copyTranscription();
        });

        // Collapsible section headers
        this.elements.collapsibleHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const target = header.dataset.target;
                this.toggleSection(target);
            });
        });

        // Listen for settings changes
        document.addEventListener('settingsChanged', (event) => {
            this.updateVisibility(event.detail.preferences);
        });

        // Listen for segment playback events
        document.addEventListener('segmentPlaybackStarted', (event) => {
            this.highlightPlayingSegment(event.detail.element, true);
        });

        document.addEventListener('segmentPlaybackStopped', (event) => {
            this.highlightPlayingSegment(event.detail.element, false);
        });

        // Listen for speaker name changes
        document.addEventListener('speakerNameChanged', (event) => {
            this.updateSpeakerLabels(event.detail.speakerId, event.detail.newName);
        });
    }

    // Collapsible functionality
    toggleSection(sectionName) {
        const section = this.container.querySelector(`[data-section="${sectionName}"]`);
        const content = this.container.querySelector(`[data-content="${sectionName}"]`);
        const indicator = this.container.querySelector(`[data-target="${sectionName}"] .collapse-indicator`);

        if (!section || !content || !indicator) return;

        const isCollapsed = this.collapsedSections.has(sectionName);

        if (isCollapsed) {
            // Expand
            this.collapsedSections.delete(sectionName);
            section.classList.remove('collapsed');
            content.style.maxHeight = content.scrollHeight + 'px';
            indicator.textContent = '▼';

            // Reset max-height after transition
            setTimeout(() => {
                if (!this.collapsedSections.has(sectionName)) {
                    content.style.maxHeight = 'none';
                }
            }, 300);
        } else {
            // Collapse
            this.collapsedSections.add(sectionName);
            section.classList.add('collapsed');
            content.style.maxHeight = content.scrollHeight + 'px';

            // Force reflow then collapse
            requestAnimationFrame(() => {
                content.style.maxHeight = '0px';
            });
            indicator.textContent = '▶';
        }

        this.saveCollapsedState();
    }

    saveCollapsedState() {
        try {
            UIUtils.saveToLocalStorage('parakeet-collapsed-sections', Array.from(this.collapsedSections));
        } catch (error) {
            console.warn('Failed to save collapsed state:', error);
        }
    }

    loadCollapsedState() {
        const saved = UIUtils.loadFromLocalStorage('parakeet-collapsed-sections', []);

        // Ensure saved data is an array before creating Set
        const savedArray = Array.isArray(saved) ? saved : [];
        this.collapsedSections = new Set(savedArray);

        // Apply collapsed state to existing sections
        this.collapsedSections.forEach(sectionName => {
            const section = this.container.querySelector(`[data-section="${sectionName}"]`);
            const content = this.container.querySelector(`[data-content="${sectionName}"]`);
            const indicator = this.container.querySelector(`[data-target="${sectionName}"] .collapse-indicator`);

            if (section && content && indicator) {
                section.classList.add('collapsed');
                content.style.maxHeight = '0px';
                indicator.textContent = '▶';
            }
        });
    }

    updateVisibility(preferences) {
        if (this.elements.segmentsSection) {
            this.elements.segmentsSection.style.display = preferences.showSegments ? 'block' : 'none';
        }

        if (this.elements.timestampsSection) {
            this.elements.timestampsSection.style.display = preferences.showWordTimestamps ? 'block' : 'none';
        }
    }

    display(data) {
        console.log('Displaying results:', data);

        // Store current data
        this.currentData = data;

        // Determine if this is diarization data
        const isDiarization = this.state.currentMode === 'diarize' && data.diarization;
        const transcriptionData = isDiarization ? data.transcription : data;

        // Display transcription
        this.displayTranscription(transcriptionData);

        // Display statistics
        this.displayStatistics(transcriptionData, data, isDiarization);

        // Display segments
        this.displaySegments(transcriptionData, isDiarization);

        // Display word timestamps
        this.displayWordTimestamps(transcriptionData);

        // Display speaker summary if applicable
        if (isDiarization) {
            this.displaySpeakerSummary(data.diarization);
        }

        // Show results
        this.show();
    }

    displayTranscription(transcriptionData) {
        let transcriptionText = '';

        if (typeof transcriptionData.text === 'string') {
            transcriptionText = transcriptionData.text;
        } else if (Array.isArray(transcriptionData.text) && transcriptionData.text.length > 0) {
            transcriptionText = transcriptionData.text[0].text || 'No transcription found';
        } else {
            transcriptionText = 'No transcription text found';
        }

        this.elements.transcriptionText.textContent = transcriptionText;
    }

    displayStatistics(transcriptionData, rawData, isDiarization) {
        // Processing time
        const processingTime = transcriptionData.processing_time_sec ||
                              rawData.combined_processing_time || 0;
        this.elements.processingTime.textContent = processingTime.toFixed(2);

        // Word count
        const text = this.getTranscriptionText(transcriptionData);
        const wordCount = text.split(' ').filter(word => word.trim().length > 0).length;
        this.elements.wordCount.textContent = wordCount.toLocaleString();

        // Segment count
        const segmentCount = transcriptionData.segments ? transcriptionData.segments.length : 0;
        this.elements.segmentCount.textContent = segmentCount.toLocaleString();

        // Speaker count (if diarization)
        if (isDiarization && rawData.diarization) {
            this.elements.speakerCount.textContent = rawData.diarization.num_speakers_detected || 0;
            this.elements.speakerCountStat.style.display = 'block';
        } else {
            this.elements.speakerCountStat.style.display = 'none';
        }
    }

    displaySegments(transcriptionData, isDiarization) {
        if (!transcriptionData.segments || transcriptionData.segments.length === 0) {
            this.elements.segments.innerHTML = '<div style="text-align: center; color: #6c757d; padding: 20px; font-style: italic;">No segments available</div>';
            return;
        }

        this.elements.segments.innerHTML = '';

        transcriptionData.segments.forEach((segment, index) => {
            const segmentElement = this.createSegmentElement(segment, index, isDiarization);
            this.elements.segments.appendChild(segmentElement);
        });
    }

    createSegmentElement(segment, index, isDiarization) {
        const segmentDiv = document.createElement('div');
        segmentDiv.className = 'segment-item';

        const startTime = segment.start || 0;
        const endTime = segment.end || 0;
        const duration = endTime - startTime;

        const speakerInfo = isDiarization && segment.speaker ?
            `<span class="speaker-label ${UIUtils.getSpeakerClass(segment.speaker)}" data-speaker="${segment.speaker}">
                ${this.state.getDisplayName(segment.speaker).toUpperCase()}
            </span>` : '';

        segmentDiv.innerHTML = `
            <div class="segment-header">
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <div class="segment-time">
                        ${UIUtils.formatTime(startTime)} - ${UIUtils.formatTime(endTime)}
                    </div>
                    ${speakerInfo}
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="segment-duration">
                        ${UIUtils.formatDuration(duration)} duration
                    </div>
                    <button class="play-button" data-start="${startTime}" data-end="${endTime}">
                        ▶
                    </button>
                </div>
            </div>
            <p class="segment-text">${segment.text}</p>
        `;

        // Add click listener for playback
        const playButton = segmentDiv.querySelector('.play-button');
        playButton.addEventListener('click', (e) => {
            e.stopPropagation();
            this.playSegment(startTime, endTime, segmentDiv);
        });

        // Add speaker label editing functionality
        if (isDiarization && segment.speaker) {
            const speakerLabel = segmentDiv.querySelector('.speaker-label');
            this.setupSpeakerLabelEditing(speakerLabel);
        }

        return segmentDiv;
    }

    displayWordTimestamps(transcriptionData) {
        const wordTimestamps = transcriptionData.timestamps?.word || [];

        if (wordTimestamps.length === 0) {
            this.elements.timestamps.innerHTML = '<div style="text-align: center; color: #6c757d; padding: 20px; font-style: italic;">No word timestamps available</div>';
            return;
        }

        this.elements.timestamps.innerHTML = '';

        wordTimestamps.forEach(wordData => {
            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'timestamp-item';

            timestampDiv.innerHTML = `
                <span class="timestamp-word">${wordData.word}</span>
                <span class="timestamp-time">${UIUtils.formatTime(wordData.start)} - ${UIUtils.formatTime(wordData.end)}</span>
            `;

            this.elements.timestamps.appendChild(timestampDiv);
        });
    }

    displaySpeakerSummary(diarizationData) {
        if (!diarizationData.speakers) {
            this.elements.speakerSummarySection.style.display = 'none';
            return;
        }

        this.elements.speakerSummary.innerHTML = '';

        Object.entries(diarizationData.speakers).forEach(([speakerId, segments]) => {
            const totalDuration = segments.reduce((sum, seg) => sum + seg.duration, 0);

            // Calculate percentage more accurately
            const allSegmentsDuration = Object.values(diarizationData.speakers)
                .flat()
                .reduce((sum, seg) => sum + seg.duration, 0);
            const percentage = allSegmentsDuration > 0 ? (totalDuration / allSegmentsDuration * 100) : 0;

            const speakerDiv = document.createElement('div');
            speakerDiv.className = 'speaker-item';

            speakerDiv.innerHTML = `
                <div class="speaker-info">
                    <span class="speaker-label ${UIUtils.getSpeakerClass(speakerId)}" data-speaker="${speakerId}">
                        ${this.state.getDisplayName(speakerId).toUpperCase()}
                    </span>
                    <span>${segments.length} segments</span>
                </div>
                <div class="speaker-stats">
                    <span>${UIUtils.formatDuration(totalDuration)}</span>
                    <span>${percentage.toFixed(1)}%</span>
                </div>
            `;

            // Add speaker label editing functionality
            const speakerLabel = speakerDiv.querySelector('.speaker-label');
            this.setupSpeakerLabelEditing(speakerLabel);

            this.elements.speakerSummary.appendChild(speakerDiv);
        });

        this.elements.speakerSummarySection.style.display = 'block';
    }

    setupSpeakerLabelEditing(speakerLabel) {
        if (!speakerLabel) return;

        const speakerId = speakerLabel.dataset.speaker;

        speakerLabel.addEventListener('click', (e) => {
            e.stopPropagation();
            this.startEditingSpeakerLabel(speakerLabel, speakerId);
        });

        speakerLabel.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.finishEditingSpeakerLabel(speakerLabel, speakerId);
            } else if (e.key === 'Escape') {
                this.cancelEditingSpeakerLabel(speakerLabel, speakerId);
            }
        });

        speakerLabel.addEventListener('blur', () => {
            this.finishEditingSpeakerLabel(speakerLabel, speakerId);
        });
    }

    startEditingSpeakerLabel(speakerLabel, speakerId) {
        if (this.currentlyEditingLabel && this.currentlyEditingLabel !== speakerLabel) {
            this.cancelEditingSpeakerLabel(this.currentlyEditingLabel, this.currentlyEditingLabel.dataset.speaker);
        }

        this.currentlyEditingLabel = speakerLabel;
        speakerLabel.classList.add('editing');
        speakerLabel.contentEditable = true;
        speakerLabel.focus();

        // Select all text
        const range = document.createRange();
        range.selectNodeContents(speakerLabel);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    }

    finishEditingSpeakerLabel(speakerLabel, speakerId) {
        if (!speakerLabel.classList.contains('editing')) return;

        const newName = speakerLabel.textContent.trim();
        if (newName && newName !== this.state.getDisplayName(speakerId)) {
            // Update state
            this.state.setSpeakerNames({
                ...this.state.speakerNames,
                [speakerId]: newName
            });

            // Update all other labels with same speaker ID
            this.updateSpeakerLabels(speakerId, newName);
        }

        speakerLabel.classList.remove('editing');
        speakerLabel.contentEditable = false;
        this.currentlyEditingLabel = null;
    }

    cancelEditingSpeakerLabel(speakerLabel, speakerId) {
        if (!speakerLabel.classList.contains('editing')) return;

        speakerLabel.textContent = this.state.getDisplayName(speakerId).toUpperCase();
        speakerLabel.classList.remove('editing');
        speakerLabel.contentEditable = false;
        this.currentlyEditingLabel = null;
    }

    updateSpeakerLabels(speakerId, newName) {
        // Update all speaker labels with the same speaker ID
        const speakerLabels = this.container.querySelectorAll(`[data-speaker="${speakerId}"]`);
        speakerLabels.forEach(label => {
            if (!label.classList.contains('editing')) {
                label.textContent = newName.toUpperCase();
            }
        });
    }

    async playSegment(startTime, endTime, segmentElement) {
        const audioInput = this.state.getAudioInput();
        if (!audioInput) {
            UIUtils.showToast('No audio available for playback', 'warning');
            return;
        }

        try {
            await this.services.audio.playAudioSegment(audioInput, startTime, endTime, segmentElement);
        } catch (error) {
            console.error('Error playing segment:', error);
            UIUtils.showToast('Could not play audio segment', 'error');
        }
    }

    highlightPlayingSegment(segmentElement, isPlaying) {
        if (segmentElement) {
            if (isPlaying) {
                segmentElement.classList.add('playing');
                const playButton = segmentElement.querySelector('.play-button');
                if (playButton) {
                    playButton.textContent = '⏸';
                    playButton.classList.add('playing');
                }
            } else {
                segmentElement.classList.remove('playing');
                const playButton = segmentElement.querySelector('.play-button');
                if (playButton) {
                    playButton.textContent = '▶';
                    playButton.classList.remove('playing');
                }
            }
        }
    }

    getTranscriptionText(transcriptionData) {
        if (typeof transcriptionData.text === 'string') {
            return transcriptionData.text;
        } else if (Array.isArray(transcriptionData.text) && transcriptionData.text.length > 0) {
            return transcriptionData.text[0].text || '';
        }
        return '';
    }

    async copyTranscription() {
        const text = this.elements.transcriptionText.textContent;
        if (text) {
            await UIUtils.copyToClipboard(text);
        } else {
            UIUtils.showToast('No transcription to copy', 'warning');
        }
    }

    show() {
        this.elements.results.style.display = 'block';

        // Apply initial settings
        const settings = this.state.components?.settings?.getUIPreferences();
        if (settings) {
            this.updateVisibility(settings);
        }

        // Apply saved collapsed states
        this.loadCollapsedState();
    }

    hide() {
        this.elements.results.style.display = 'none';
        this.currentData = null;

        // Cancel any ongoing edits
        if (this.currentlyEditingLabel) {
            this.cancelEditingSpeakerLabel(
                this.currentlyEditingLabel,
                this.currentlyEditingLabel.dataset.speaker
            );
        }
    }

    // Public method to refresh display with current data
    refresh() {
        if (this.currentData) {
            this.display(this.currentData);
        }
    }

    // Get current results data for download
    getCurrentData() {
        return this.currentData;
    }
}
