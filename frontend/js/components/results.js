// Results Component - Displays transcription results with speaker support
class ResultsComponent {
    constructor(state, services) {
        this.state = state;
        this.services = services;
        this.container = document.getElementById('results-component');
        this.currentlyEditingLabel = null;
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="results" id="results" style="display: none;">
                <!-- Main Transcription -->
                <div class="result-section">
                    <h3>📝 Transcription</h3>
                    <div class="transcription-text" id="transcription-text"></div>
                    <button class="btn btn-secondary" id="copy-transcription-btn" style="margin-top: 10px;">
                        📋 Copy Text
                    </button>
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

                <!-- Speaker Summary -->
                <div class="result-section" id="speaker-summary-section" style="display: none;">
                    <h3>👥 Speaker Summary <span class="subtitle">Click names to edit</span></h3>
                    <div class="speaker-summary" id="speaker-summary"></div>
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
            timestamps: this.container.querySelector('#timestamps')
        };
    }

    setupEventListeners() {
        // Copy transcription button
        this.elements.copyTranscriptionBtn.addEventListener('click', () => {
            this.copyTranscription();
        });

        // Audio service events
        document.addEventListener('segmentPlaybackStarted', (event) => {
            this.updateSegmentPlaybackUI(event.detail.element, true);
        });

        document.addEventListener('segmentPlaybackStopped', (event) => {
            this.updateSegmentPlaybackUI(event.detail.element, false);
        });

        // Handle clicks outside editing labels
        document.addEventListener('click', (e) => {
            if (this.currentlyEditingLabel && !this.currentlyEditingLabel.contains(e.target)) {
                this.finishEditingSpeaker(this.currentlyEditingLabel, true);
            }
        });
    }

    // Main display method
    display(data) {
        console.log('Displaying results:', data);

        // Reset state
        this.hide();

        // Extract data based on format
        let transcriptionData, diarizationData;
        const mode = this.state.currentMode;

        if (mode === 'diarize' && data.transcription) {
            transcriptionData = data.transcription;
            diarizationData = data.diarization;
        } else {
            transcriptionData = data;
            diarizationData = null;
        }

        // Display transcription
        this.displayTranscription(transcriptionData);

        // Display statistics
        this.displayStatistics(transcriptionData, diarizationData, data);

        // Display speaker information
        if (diarizationData) {
            this.initializeSpeakerNames(diarizationData.speakers);
            this.displaySpeakerSummary(diarizationData);
        }

        // Display segments and timestamps based on settings
        const preferences = window.app.components.settings.getUIPreferences();
        this.displaySegments(transcriptionData, preferences, mode === 'diarize');
        this.displayTimestamps(transcriptionData, preferences);

        // Show results
        this.show();
    }

    displayTranscription(data) {
        let transcriptionText = '';

        if (typeof data.text === 'string') {
            transcriptionText = data.text;
        } else if (Array.isArray(data.text) && data.text.length > 0) {
            transcriptionText = data.text[0].text || 'No transcription found';
        } else {
            transcriptionText = 'No transcription text found';
        }

        this.elements.transcriptionText.textContent = transcriptionText;
    }

    displayStatistics(transcriptionData, diarizationData, rawData) {
        // Processing time
        const processingTime = transcriptionData.processing_time_sec ||
                             rawData.combined_processing_time || 0;
        this.elements.processingTime.textContent = processingTime.toFixed(2);

        // Word count
        const text = typeof transcriptionData.text === 'string'
            ? transcriptionData.text
            : (transcriptionData.text?.[0]?.text || '');
        const wordCount = text.split(' ').filter(word => word.trim().length > 0).length;
        this.elements.wordCount.textContent = wordCount;

        // Segment count
        const segmentCount = transcriptionData.segments?.length || 0;
        this.elements.segmentCount.textContent = segmentCount;

        // Speaker count (if available)
        if (diarizationData) {
            const speakerCount = diarizationData.num_speakers_detected || 0;
            this.elements.speakerCount.textContent = speakerCount;
            this.elements.speakerCountStat.style.display = 'block';
        } else {
            this.elements.speakerCountStat.style.display = 'none';
        }
    }

    displaySpeakerSummary(diarizationData) {
        if (!diarizationData.speakers) {
            this.elements.speakerSummarySection.style.display = 'none';
            return;
        }

        UIUtils.clearElement(this.elements.speakerSummary);

        // Calculate speaker statistics
        const speakerStats = this.calculateSpeakerStats(diarizationData.speakers);

        // Display each speaker
        speakerStats.forEach(stat => {
            const speakerItem = this.createSpeakerSummaryItem(stat);
            this.elements.speakerSummary.appendChild(speakerItem);
        });

        this.elements.speakerSummarySection.style.display = 'block';
    }

    calculateSpeakerStats(speakers) {
        const stats = [];
        let totalDuration = 0;

        Object.entries(speakers).forEach(([speaker, segments]) => {
            const duration = segments.reduce((sum, seg) => sum + seg.duration, 0);
            totalDuration += duration;
            stats.push({
                speaker: speaker,
                duration: duration,
                segments: segments.length
            });
        });

        // Sort by duration and add percentages
        stats.sort((a, b) => b.duration - a.duration);
        stats.forEach(stat => {
            stat.percentage = ((stat.duration / totalDuration) * 100).toFixed(1);
        });

        return stats;
    }

    createSpeakerSummaryItem(stat) {
        const speakerItem = UIUtils.createElement('div', 'speaker-item');

        const speakerInfo = UIUtils.createElement('div', 'speaker-info');
        const speakerLabel = this.createEditableSpeakerLabel(stat.speaker);
        const segmentInfo = UIUtils.createElement('span', '', `${stat.segments} segments`);

        speakerInfo.appendChild(speakerLabel);
        speakerInfo.appendChild(segmentInfo);

        const speakerStatsDiv = UIUtils.createElementWithHTML('div', 'speaker-stats', `
            <span>${stat.duration.toFixed(1)}s</span>
            <span>${stat.percentage}%</span>
        `);

        speakerItem.appendChild(speakerInfo);
        speakerItem.appendChild(speakerStatsDiv);

        return speakerItem;
    }

    displaySegments(data, preferences, showSpeakers) {
        if (!preferences.showSegments || !data.segments || data.segments.length === 0) {
            this.elements.segmentsSection.style.display = 'none';
            return;
        }

        UIUtils.clearElement(this.elements.segments);

        data.segments.forEach((segment, index) => {
            const segmentItem = this.createSegmentItem(segment, index, showSpeakers);
            this.elements.segments.appendChild(segmentItem);
        });

        this.elements.segmentsSection.style.display = 'block';
    }

    createSegmentItem(segment, index, showSpeakers) {
        const segmentItem = UIUtils.createElement('div', 'segment-item');

        // Extract segment data
        const startTime = segment.start || 0;
        const endTime = segment.end || 0;
        const duration = (endTime - startTime).toFixed(2);
        const text = segment.text || segment.segment || 'No text available';
        const speaker = segment.speaker;

        // Create header
        const header = UIUtils.createElement('div', 'segment-header');

        const leftSection = UIUtils.createElement('div', '');
        leftSection.style.cssText = 'display: flex; align-items: center; gap: 10px;';

        // Play button
        const playButton = UIUtils.createElement('button', 'play-button', '▶️');
        playButton.title = 'Play this segment';

        // Time display
        const timeDiv = UIUtils.createElement('div', 'segment-time',
            `${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`);

        leftSection.appendChild(playButton);
        leftSection.appendChild(timeDiv);

        // Speaker label
        if (showSpeakers && speaker) {
            const speakerLabel = this.createEditableSpeakerLabel(speaker);
            leftSection.appendChild(speakerLabel);
        }

        // Duration
        const durationDiv = UIUtils.createElement('div', 'segment-duration', `${duration}s duration`);

        header.appendChild(leftSection);
        header.appendChild(durationDiv);

        // Segment text
        const textDiv = UIUtils.createElement('div', 'segment-text', text);

        segmentItem.appendChild(header);
        segmentItem.appendChild(textDiv);

        // Add click handlers for playback
        const playHandler = async (e) => {
            e.stopPropagation();

            if (this.services.audio.currentlyPlayingSegment === segmentItem) {
                this.services.audio.stopCurrentAudio();
            } else {
                try {
                    const audioInput = this.state.getAudioInput();
                    await this.services.audio.playAudioSegment(
                        audioInput, startTime, endTime, segmentItem
                    );
                } catch (error) {
                    console.error('Error playing segment:', error);
                    UIUtils.showToast('Could not play audio segment', 'error');
                }
            }
        };

        playButton.addEventListener('click', playHandler);
        segmentItem.addEventListener('click', playHandler);

        return segmentItem;
    }

    displayTimestamps(data, preferences) {
        if (!preferences.showWordTimestamps ||
            !data.timestamps?.word ||
            data.timestamps.word.length === 0) {
            this.elements.timestampsSection.style.display = 'none';
            return;
        }

        UIUtils.clearElement(this.elements.timestamps);

        data.timestamps.word.forEach(item => {
            const timestampItem = UIUtils.createElementWithHTML('div', 'timestamp-item', `
                <span class="timestamp-word">${item.word}</span>
                <span class="timestamp-time">${item.start.toFixed(2)}s - ${item.end.toFixed(2)}s</span>
            `);

            this.elements.timestamps.appendChild(timestampItem);
        });

        this.elements.timestampsSection.style.display = 'block';
    }

    // Speaker management
    initializeSpeakerNames(speakers) {
        this.state.setSpeakerNames(UIUtils.initializeSpeakerNames(speakers));
    }

    createEditableSpeakerLabel(speakerId) {
        const displayName = this.state.getDisplayName(speakerId);
        const speakerClass = UIUtils.getSpeakerClass(speakerId);

        const label = UIUtils.createElementWithHTML('div',
            `speaker-label editable ${speakerClass}`, `
            ${displayName.toUpperCase()}
            <div class="speaker-rename-hint">Click to rename</div>
        `);

        label.setAttribute('data-speaker-id', speakerId);

        label.addEventListener('click', (e) => {
            e.stopPropagation();
            this.startEditingSpeaker(label, speakerId);
        });

        return label;
    }

    startEditingSpeaker(labelElement, speakerId) {
        if (this.currentlyEditingLabel) {
            this.finishEditingSpeaker(this.currentlyEditingLabel, false);
        }

        this.currentlyEditingLabel = labelElement;
        const currentName = this.state.getDisplayName(speakerId);

        labelElement.classList.add('editing');

        const input = UIUtils.createElement('input', 'speaker-label-input');
        input.type = 'text';
        input.value = currentName;
        input.maxLength = 20;

        labelElement.innerHTML = '';
        labelElement.appendChild(input);

        input.focus();
        input.select();

        const finishEdit = (save = true) => {
            this.finishEditingSpeaker(labelElement, save, speakerId, input.value.trim());
        };

        input.addEventListener('blur', () => finishEdit(true));
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') finishEdit(true);
            else if (e.key === 'Escape') finishEdit(false);
        });
        input.addEventListener('click', (e) => e.stopPropagation());
    }

    finishEditingSpeaker(labelElement, save = true, speakerId = null, newName = null) {
        if (!labelElement || !labelElement.classList.contains('editing')) return;

        labelElement.classList.remove('editing');
        this.currentlyEditingLabel = null;

        if (save && speakerId && newName && newName.length > 0) {
            // Update speaker name in state
            const newSpeakerNames = { ...this.state.speakerNames };
            newSpeakerNames[speakerId] = newName;
            this.state.setSpeakerNames(newSpeakerNames);

            // Update all labels with this speaker ID
            this.updateAllSpeakerLabels(speakerId, newName);
        } else {
            // Restore original content
            const currentName = speakerId ? this.state.getDisplayName(speakerId) : 'Unknown';
            labelElement.innerHTML = `
                ${currentName.toUpperCase()}
                <div class="speaker-rename-hint">Click to rename</div>
            `;
        }
    }

    updateAllSpeakerLabels(speakerId, newName) {
        const allLabels = document.querySelectorAll('.speaker-label');
        allLabels.forEach(label => {
            const labelSpeakerId = label.getAttribute('data-speaker-id');
            if (labelSpeakerId === speakerId && !label.classList.contains('editing')) {
                label.innerHTML = `
                    ${newName.toUpperCase()}
                    <div class="speaker-rename-hint">Click to rename</div>
                `;
            }
        });
    }

    updateSegmentPlaybackUI(segmentElement, isPlaying) {
        if (isPlaying) {
            segmentElement.classList.add('playing');
            const playButton = segmentElement.querySelector('.play-button');
            if (playButton) {
                playButton.textContent = '⏸️';
                playButton.classList.add('playing');
            }
        } else {
            segmentElement.classList.remove('playing');
            const playButton = segmentElement.querySelector('.play-button');
            if (playButton) {
                playButton.textContent = '▶️';
                playButton.classList.remove('playing');
            }
        }
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
        UIUtils.fadeIn(this.elements.results);
    }

    hide() {
        this.elements.results.style.display = 'none';
    }
}
