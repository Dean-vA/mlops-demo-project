// Complete Results Component - Full functionality from monolithic version
class ResultsComponent {
    constructor(state, services) {
        this.state = state;
        this.services = services;
        this.container = document.getElementById('results-component');
        this.currentlyEditingLabel = null;
        this.currentData = null;
        this.currentlyPlayingSegment = null;
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
                    <div class="segments custom-scrollbar" id="segments"></div>
                </div>

                <!-- Word Timestamps -->
                <div class="result-section" id="timestamps-section" style="display: none;">
                    <h3>⏱️ Word Timestamps</h3>
                    <div class="timestamps custom-scrollbar" id="timestamps"></div>
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

        // Listen for settings changes to update visibility
        document.addEventListener('settingsChanged', (event) => {
            console.log('ResultsComponent received settingsChanged event:', event.detail);
            if (event.detail && event.detail.preferences) {
                const preferences = event.detail.preferences;
                console.log('Applying preferences:', preferences);
                this.updateVisibility(preferences);
            }
        });

        // Listen for audio playback events
        document.addEventListener('segmentPlaybackStarted', (event) => {
            const { element } = event.detail;
            if (element) {
                element.classList.add('playing');
                const playButton = element.querySelector('.play-button');
                if (playButton) {
                    playButton.textContent = '⏸️';
                    playButton.classList.add('playing');
                }
            }
        });

        document.addEventListener('segmentPlaybackStopped', (event) => {
            const { element } = event.detail;
            if (element) {
                element.classList.remove('playing');
                const playButton = element.querySelector('.play-button');
                if (playButton) {
                    playButton.textContent = '▶️';
                    playButton.classList.remove('playing');
                }
            }
        });

        // Handle clicks outside of editing labels to finish editing
        document.addEventListener('click', (e) => {
            if (this.currentlyEditingLabel && !this.currentlyEditingLabel.contains(e.target)) {
                this.finishEditing(this.currentlyEditingLabel, true);
            }
        });
    }

    // Main display method
    display(data) {
        console.log('Results display() called with data:', data);

        // Store current data
        this.currentData = data;

        // Reset state
        this.hide();

        // Determine mode and data structure
        const isDiarizeMode = this.state.currentMode === 'diarize' && data.transcription;
        const transcriptionData = isDiarizeMode ? data.transcription : data;
        const diarizationData = isDiarizeMode ? data.diarization : null;

        console.log('isDiarizeMode:', isDiarizeMode);
        console.log('transcriptionData:', transcriptionData);
        console.log('diarizationData:', diarizationData);

        // Display transcription text
        this.displayTranscriptionText(transcriptionData);

        // Display statistics
        this.displayStatistics(transcriptionData, data, diarizationData);

        // Display speaker information (if available)
        if (isDiarizeMode && diarizationData) {
            this.initializeSpeakerNames(diarizationData.speakers);
            this.displaySpeakerSummary(diarizationData);
        }

        // Display segments and timestamps - check settings before showing
        this.displaySegments(transcriptionData, isDiarizeMode ? true : false);
        this.displayTimestamps(transcriptionData);

        // Apply current settings after displaying
        const preferences = window.app?.components?.settings?.getUIPreferences();
        if (preferences) {
            this.updateVisibility(preferences);
        }

        // Show results
        this.show();
    }

    displayTranscriptionText(transcriptionData) {
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

    displayStatistics(transcriptionData, rawData, diarizationData) {
        // Processing time
        const processingTime = transcriptionData.processing_time_sec ||
                              rawData.combined_processing_time || 0;
        this.elements.processingTime.textContent = processingTime.toFixed(2);

        // Word count
        const text = this.getTextContent(transcriptionData);
        const words = text.split(' ').filter(word => word.trim().length > 0).length;
        this.elements.wordCount.textContent = words;

        // Segment count
        const segments = this.getSegments(transcriptionData);
        this.elements.segmentCount.textContent = segments.length;

        // Speaker count (if diarization data available)
        if (diarizationData) {
            this.elements.speakerCount.textContent = diarizationData.num_speakers_detected || 0;
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

        this.elements.speakerSummary.innerHTML = '';

        // Calculate speaker statistics
        const speakerStats = [];
        let totalDuration = 0;

        Object.entries(diarizationData.speakers).forEach(([speaker, segments]) => {
            const duration = segments.reduce((sum, seg) => sum + seg.duration, 0);
            totalDuration += duration;
            speakerStats.push({
                speaker: speaker,
                duration: duration,
                segments: segments.length
            });
        });

        // Sort by duration (most active first)
        speakerStats.sort((a, b) => b.duration - a.duration);

        // Display each speaker with editable labels
        speakerStats.forEach(stat => {
            const percentage = ((stat.duration / totalDuration) * 100).toFixed(1);

            const speakerItem = document.createElement('div');
            speakerItem.className = 'speaker-item';

            const speakerInfoDiv = document.createElement('div');
            speakerInfoDiv.className = 'speaker-info';

            // Create editable speaker label
            const speakerLabel = this.createEditableSpeakerLabel(stat.speaker);
            speakerInfoDiv.appendChild(speakerLabel);

            const segmentsSpan = document.createElement('span');
            segmentsSpan.textContent = `${stat.segments} segments`;
            speakerInfoDiv.appendChild(segmentsSpan);

            const speakerStatsDiv = document.createElement('div');
            speakerStatsDiv.className = 'speaker-stats';
            speakerStatsDiv.innerHTML = `
                <span>${stat.duration.toFixed(1)}s</span>
                <span>${percentage}%</span>
            `;

            speakerItem.appendChild(speakerInfoDiv);
            speakerItem.appendChild(speakerStatsDiv);
            this.elements.speakerSummary.appendChild(speakerItem);
        });

        this.elements.speakerSummarySection.style.display = 'block';
    }

    displaySegments(transcriptionData, showSpeakers = false) {
        const segments = this.getSegments(transcriptionData);
        this.elements.segments.innerHTML = '';

        console.log('displaySegments called with', segments.length, 'segments, showSpeakers:', showSpeakers);

        if (segments.length === 0) {
            console.log('No segments to display, hiding section');
            this.elements.segmentsSection.style.display = 'none';
            return;
        }

        segments.forEach((segment, index) => {
            const segmentItem = document.createElement('div');
            segmentItem.className = 'segment-item';

            // Extract segment data with fallbacks for different formats
            const segmentStart = segment.start || 0;
            const segmentEnd = segment.end || 0;
            const segmentText = segment.text || segment.segment || 'No text available';
            const segmentSpeaker = segment.speaker;
            const duration = (segmentEnd - segmentStart).toFixed(2);

            // Create segment header
            const segmentHeader = document.createElement('div');
            segmentHeader.className = 'segment-header';

            const leftSection = document.createElement('div');
            leftSection.style.cssText = 'display: flex; align-items: center; gap: 10px; flex-wrap: wrap;';

            // Play button
            const playButton = document.createElement('button');
            playButton.className = 'play-button';
            playButton.title = 'Play this segment';
            playButton.textContent = '▶️';

            // Time display
            const timeDiv = document.createElement('div');
            timeDiv.className = 'segment-time';
            timeDiv.textContent = `${segmentStart.toFixed(2)}s - ${segmentEnd.toFixed(2)}s`;

            leftSection.appendChild(playButton);
            leftSection.appendChild(timeDiv);

            // Add speaker label if showing speakers and speaker info exists
            if (showSpeakers && segmentSpeaker) {
                const speakerLabel = this.createEditableSpeakerLabel(segmentSpeaker);
                leftSection.appendChild(speakerLabel);
            }

            // Duration display
            const durationDiv = document.createElement('div');
            durationDiv.className = 'segment-duration';
            durationDiv.textContent = `${duration}s duration`;

            segmentHeader.appendChild(leftSection);
            segmentHeader.appendChild(durationDiv);

            // Segment text
            const segmentTextDiv = document.createElement('div');
            segmentTextDiv.className = 'segment-text';
            segmentTextDiv.textContent = segmentText;

            segmentItem.appendChild(segmentHeader);
            segmentItem.appendChild(segmentTextDiv);

            // Add click handlers for audio playback
            const playHandler = (e) => {
                e.stopPropagation();
                this.handleSegmentPlayback(segmentStart, segmentEnd, segmentItem);
            };

            playButton.addEventListener('click', playHandler);
            segmentItem.addEventListener('click', playHandler);

            this.elements.segments.appendChild(segmentItem);
        });

        console.log(`Successfully displayed ${segments.length} segments`);

        // Force show the section
        this.elements.segmentsSection.style.display = 'block';
    }

    displayTimestamps(transcriptionData) {
        const wordTimestamps = this.getWordTimestamps(transcriptionData);
        this.elements.timestamps.innerHTML = '';

        if (wordTimestamps.length === 0) {
            this.elements.timestampsSection.style.display = 'none';
            return;
        }

        wordTimestamps.forEach(item => {
            const timestampItem = document.createElement('div');
            timestampItem.className = 'timestamp-item';

            const wordSpan = document.createElement('span');
            wordSpan.className = 'timestamp-word';
            wordSpan.textContent = item.word || '';

            const timeSpan = document.createElement('span');
            timeSpan.className = 'timestamp-time';
            const start = (item.start || 0).toFixed(2);
            const end = (item.end || 0).toFixed(2);
            timeSpan.textContent = `${start}s - ${end}s`;

            timestampItem.appendChild(wordSpan);
            timestampItem.appendChild(timeSpan);

            this.elements.timestamps.appendChild(timestampItem);
        });

        // Force show the section
        this.elements.timestampsSection.style.display = 'block';
    }

    // Speaker management methods
    initializeSpeakerNames(speakers) {
        if (!speakers) return;

        // Create default friendly names for each speaker
        Object.keys(speakers).forEach((speakerId, index) => {
            if (speakerId === 'unknown') {
                this.state.speakerNames[speakerId] = 'Unknown';
            } else {
                this.state.speakerNames[speakerId] = `Speaker ${index + 1}`;
            }
        });

        this.state.setSpeakerNames(this.state.speakerNames);
    }

    createEditableSpeakerLabel(speakerId, additionalClasses = '') {
        const displayName = this.state.getDisplayName(speakerId);
        const speakerClass = UIUtils.getSpeakerClass(speakerId);

        const label = document.createElement('div');
        label.className = `speaker-label editable ${speakerClass} ${additionalClasses}`;
        label.setAttribute('data-speaker-id', speakerId);
        label.innerHTML = `
            ${displayName.toUpperCase()}
            <div class="speaker-rename-hint">Click to rename</div>
        `;

        // Add click handler for editing
        label.addEventListener('click', (e) => {
            e.stopPropagation();
            this.makeEditable(label, speakerId);
        });

        return label;
    }

    makeEditable(labelElement, originalSpeakerId) {
        if (this.currentlyEditingLabel) {
            this.finishEditing(this.currentlyEditingLabel, false);
        }

        this.currentlyEditingLabel = labelElement;
        const currentName = this.state.getDisplayName(originalSpeakerId);

        // Add editing class
        labelElement.classList.add('editing');

        // Create input element
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentName;
        input.className = 'speaker-label-input';
        input.maxLength = 20;

        // Replace content with input
        labelElement.innerHTML = '';
        labelElement.appendChild(input);

        // Focus and select text
        input.focus();
        input.select();

        // Handle input events
        const finishEdit = (save = true) => {
            this.finishEditing(labelElement, save, originalSpeakerId, input.value.trim());
        };

        input.addEventListener('blur', () => finishEdit(true));
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                finishEdit(true);
            } else if (e.key === 'Escape') {
                finishEdit(false);
            }
        });

        // Prevent label click from bubbling
        input.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    finishEditing(labelElement, save = true, originalSpeakerId = null, newName = null) {
        if (!labelElement || !labelElement.classList.contains('editing')) return;

        labelElement.classList.remove('editing');
        this.currentlyEditingLabel = null;

        if (save && originalSpeakerId && newName && newName.length > 0) {
            // Update speaker name in state
            const updatedNames = { ...this.state.speakerNames };
            updatedNames[originalSpeakerId] = newName;
            this.state.setSpeakerNames(updatedNames);

            // Update this label
            labelElement.innerHTML = `
                ${newName.toUpperCase()}
                <div class="speaker-rename-hint">Click to rename</div>
            `;

            // Update all other labels with the same speaker ID
            this.updateAllSpeakerLabels(originalSpeakerId, newName);
        } else {
            // Restore original content
            const currentName = originalSpeakerId ? this.state.getDisplayName(originalSpeakerId) : labelElement.textContent;
            labelElement.innerHTML = `
                ${currentName.toUpperCase()}
                <div class="speaker-rename-hint">Click to rename</div>
            `;
        }
    }

    updateAllSpeakerLabels(originalSpeakerId, newName) {
        // Update all speaker labels throughout the interface
        const allLabels = document.querySelectorAll('.speaker-label');

        allLabels.forEach(label => {
            const speakerId = label.getAttribute('data-speaker-id');
            if (speakerId === originalSpeakerId && !label.classList.contains('editing')) {
                label.innerHTML = `
                    ${newName.toUpperCase()}
                    <div class="speaker-rename-hint">Click to rename</div>
                `;
            }
        });
    }

    // Audio playback handling
    async handleSegmentPlayback(startTime, endTime, segmentElement) {
        const audioInput = this.state.getAudioInput();
        if (!audioInput) {
            UIUtils.showToast('Audio not available for playback', 'error');
            return;
        }

        try {
            // Stop current playback if any
            if (this.currentlyPlayingSegment === segmentElement) {
                this.services.audio.stopCurrentAudio();
                return;
            }

            // Play the segment
            await this.services.audio.playAudioSegment(audioInput, startTime, endTime, segmentElement);
            this.currentlyPlayingSegment = segmentElement;

        } catch (error) {
            console.error('Error playing audio segment:', error);
            UIUtils.showToast('Could not play audio segment', 'error');
        }
    }

    // Utility methods
    getTextContent(transcriptionData) {
        if (typeof transcriptionData.text === 'string') {
            return transcriptionData.text;
        } else if (Array.isArray(transcriptionData.text) && transcriptionData.text.length > 0) {
            return transcriptionData.text[0].text || '';
        }
        return '';
    }

    getSegments(transcriptionData) {
        console.log('getSegments called with:', transcriptionData);

        // Try multiple possible locations for segments
        let segments = [];

        // Method 1: Direct segments array (clean format)
        if (transcriptionData.segments && Array.isArray(transcriptionData.segments)) {
            segments = transcriptionData.segments;
            console.log('Found segments in transcriptionData.segments:', segments.length);
        }
        // Method 2: Timestamps.segment
        else if (transcriptionData.timestamps && Array.isArray(transcriptionData.timestamps.segment)) {
            segments = transcriptionData.timestamps.segment;
            console.log('Found segments in transcriptionData.timestamps.segment:', segments.length);
        }
        // Method 3: Legacy NeMo format
        else if (Array.isArray(transcriptionData.text) && transcriptionData.text[0]?.timestamp?.segment) {
            segments = transcriptionData.text[0].timestamp.segment;
            console.log('Found segments in legacy format:', segments.length);
        }
        // Method 4: Check if text array has segments directly
        else if (Array.isArray(transcriptionData.text)) {
            for (const textItem of transcriptionData.text) {
                if (textItem.segments && Array.isArray(textItem.segments)) {
                    segments = textItem.segments;
                    console.log('Found segments in text item:', segments.length);
                    break;
                }
            }
        }

        console.log('Final segments found:', segments.length);
        if (segments.length > 0) {
            console.log('Sample segment:', segments[0]);
        }

        return segments;
    }

    getWordTimestamps(transcriptionData) {
        console.log('getWordTimestamps called with:', transcriptionData);

        let wordTimestamps = [];

        // Method 1: Direct timestamps.word array (clean format)
        if (transcriptionData.timestamps && Array.isArray(transcriptionData.timestamps.word)) {
            wordTimestamps = transcriptionData.timestamps.word;
            console.log('Found word timestamps in transcriptionData.timestamps.word:', wordTimestamps.length);
        }
        // Method 2: Legacy NeMo format
        else if (Array.isArray(transcriptionData.text) && transcriptionData.text[0]?.timestamp?.word) {
            wordTimestamps = transcriptionData.text[0].timestamp.word;
            console.log('Found word timestamps in legacy format:', wordTimestamps.length);
        }
        // Method 3: Check if text array has word timestamps directly
        else if (Array.isArray(transcriptionData.text)) {
            for (const textItem of transcriptionData.text) {
                if (textItem.timestamps && Array.isArray(textItem.timestamps.word)) {
                    wordTimestamps = textItem.timestamps.word;
                    console.log('Found word timestamps in text item:', wordTimestamps.length);
                    break;
                }
            }
        }

        console.log('Final word timestamps found:', wordTimestamps.length);
        if (wordTimestamps.length > 0) {
            console.log('Sample word timestamp:', wordTimestamps[0]);
        }

        return wordTimestamps;
    }

    updateVisibility(preferences) {
        console.log('updateVisibility called with:', preferences);
        if (!preferences) return;

        console.log('Setting segments visibility to:', preferences.showSegments);
        console.log('Setting timestamps visibility to:', preferences.showWordTimestamps);

        // Update segments visibility
        if (preferences.showSegments) {
            this.elements.segmentsSection.style.display = 'block';
        } else {
            this.elements.segmentsSection.style.display = 'none';
        }

        // Update timestamps visibility
        if (preferences.showWordTimestamps) {
            this.elements.timestampsSection.style.display = 'block';
        } else {
            this.elements.timestampsSection.style.display = 'none';
        }
    }

    getTranscriptionData() {
        if (!this.currentData) return {};

        const isDiarizeMode = this.state.currentMode === 'diarize' && this.currentData.transcription;
        return isDiarizeMode ? this.currentData.transcription : this.currentData;
    }

    async copyTranscription() {
        const text = this.elements.transcriptionText.textContent;
        if (text) {
            const success = await UIUtils.copyToClipboard(text);
            if (success) {
                UIUtils.showToast('Transcription copied to clipboard!', 'success');
            }
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
        this.currentData = null;
        this.currentlyEditingLabel = null;
        this.currentlyPlayingSegment = null;
    }
}
