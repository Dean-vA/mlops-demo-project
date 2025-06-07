// Debug Results Component - Let's find the exact issue
class ResultsComponent {
    constructor(state, services) {
        this.state = state;
        this.services = services;
        this.container = document.getElementById('results-component');
        this.currentlyEditingLabel = null;
        this.currentData = null;
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
                <div class="result-section" id="segments-section" style="display: block; border: 2px solid red;">
                    <h3>📑 Segments <span class="subtitle">Click to play audio</span></h3>
                    <div class="segments" id="segments" style="border: 2px solid blue; min-height: 50px;">
                        <div style="color: red; padding: 10px;">DEBUG: Segments container</div>
                    </div>
                </div>

                <!-- Word Timestamps -->
                <div class="result-section" id="timestamps-section" style="display: block; border: 2px solid green;">
                    <h3>⏱️ Word Timestamps</h3>
                    <div class="timestamps" id="timestamps" style="border: 2px solid orange; min-height: 50px;">
                        <div style="color: green; padding: 10px;">DEBUG: Timestamps container</div>
                    </div>
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

        console.log('DEBUG: Elements cached:', {
            segmentsSection: !!this.elements.segmentsSection,
            segments: !!this.elements.segments,
            timestampsSection: !!this.elements.timestampsSection,
            timestamps: !!this.elements.timestamps
        });
    }

    setupEventListeners() {
        // Copy transcription button
        this.elements.copyTranscriptionBtn.addEventListener('click', () => {
            this.copyTranscription();
        });

        // Listen for settings changes to update visibility
        document.addEventListener('settingsChanged', (event) => {
            console.log('DEBUG: Settings changed event received:', event.detail);
            this.testToggleNow(event.detail.preferences);
        });
    }

    // TEST: Direct toggle test
    testToggleNow(preferences) {
        console.log('DEBUG: testToggleNow called with:', preferences);
        console.log('DEBUG: Segments section element:', this.elements.segmentsSection);
        console.log('DEBUG: Timestamps section element:', this.elements.timestampsSection);

        if (this.elements.segmentsSection) {
            console.log('DEBUG: Setting segments display to:', preferences.showSegments ? 'block' : 'none');
            this.elements.segmentsSection.style.display = preferences.showSegments ? 'block' : 'none';
        }

        if (this.elements.timestampsSection) {
            console.log('DEBUG: Setting timestamps display to:', preferences.showWordTimestamps ? 'block' : 'none');
            this.elements.timestampsSection.style.display = preferences.showWordTimestamps ? 'block' : 'none';
        }

        // Force a visual check
        setTimeout(() => {
            console.log('DEBUG: After toggle - segments visible:',
                this.elements.segmentsSection.style.display !== 'none');
            console.log('DEBUG: After toggle - timestamps visible:',
                this.elements.timestampsSection.style.display !== 'none');
        }, 100);
    }

    // Main display method
    display(data) {
        console.log('DEBUG: display() called with data:', data);

        // Store current data
        this.currentData = data;

        // Reset state
        this.hide();

        // Display transcription
        this.displayTranscription(data);

        // Show results
        this.show();

        console.log('DEBUG: Results displayed, sections should be visible');
    }

    displayTranscription(data) {
        let transcriptionText = '';

        if (typeof data.text === 'string') {
            transcriptionText = data.text;
        } else if (Array.isArray(data.text) && data.text.length > 0) {
            transcriptionText = data.text[0].text || 'No transcription found';
        } else if (data.transcription && typeof data.transcription.text === 'string') {
            transcriptionText = data.transcription.text;
        } else {
            transcriptionText = 'No transcription text found';
        }

        this.elements.transcriptionText.textContent = transcriptionText;
        console.log('DEBUG: Transcription text set:', transcriptionText.substring(0, 100) + '...');
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
        console.log('DEBUG: Results container shown');
    }

    hide() {
        this.elements.results.style.display = 'none';
        this.currentData = null;
    }
}
