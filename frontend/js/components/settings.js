// Settings Component - Handles user preferences and transcription settings
class SettingsComponent {
    constructor(state) {
        this.state = state;
        this.container = document.getElementById('settings-component');
        this.settings = {
            returnTimestamps: true, // Always true for API
            showSegments: true,
            showWordTimestamps: true,
            numSpeakers: null,
            chunkDurationSec: null,
            overlapDurationSec: null
        };
    }

    async init() {
        this.render();
        this.setupEventListeners();
        this.loadSettings();

        // Emit initial settings to set up the UI correctly
        setTimeout(() => {
            this.emitSettingsChange();
        }, 100);
    }

    render() {
        this.container.innerHTML = `
            <div class="controls">
                <div class="control-group">
                    <label for="segments-toggle">Show Segment Breakdown</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="segments-toggle" checked>
                        <span class="slider"></span>
                    </label>
                </div>

                <div class="control-group">
                    <label for="timestamps-toggle">Include Word Timestamps</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="timestamps-toggle" checked>
                        <span class="slider"></span>
                    </label>
                </div>

                <div class="control-group">
                    <label for="num-speakers">Number of Speakers (optional)</label>
                    <input type="number" id="num-speakers" class="form-control"
                           min="1" max="20" placeholder="Auto-detect">
                    <small class="form-text">Leave empty for automatic detection</small>
                </div>

                <div class="control-group">
                    <label for="chunk-duration">Chunk Duration (seconds)</label>
                    <input type="number" id="chunk-duration" class="form-control"
                           min="10" max="300" placeholder="Default">
                    <small class="form-text">Advanced: Override default audio chunking</small>
                </div>

                <div class="control-group">
                    <label for="overlap-duration">Overlap Duration (seconds)</label>
                    <input type="number" id="overlap-duration" class="form-control"
                           min="0" max="30" placeholder="Default">
                    <small class="form-text">Advanced: Overlap between chunks</small>
                </div>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            segmentsToggle: this.container.querySelector('#segments-toggle'),
            timestampsToggle: this.container.querySelector('#timestamps-toggle'),
            numSpeakersInput: this.container.querySelector('#num-speakers'),
            chunkDurationInput: this.container.querySelector('#chunk-duration'),
            overlapDurationInput: this.container.querySelector('#overlap-duration')
        };
    }

    setupEventListeners() {
        // Toggle switches
        this.elements.segmentsToggle.addEventListener('change', () => {
            this.settings.showSegments = this.elements.segmentsToggle.checked;
            this.saveSettings();
            this.emitSettingsChange();
            console.log('DEBUG: Segments toggle changed to:', this.settings.showSegments);
        });

        this.elements.timestampsToggle.addEventListener('change', () => {
            this.settings.showWordTimestamps = this.elements.timestampsToggle.checked;
            this.saveSettings();
            this.emitSettingsChange();
            console.log('DEBUG: Timestamps toggle changed to:', this.settings.showWordTimestamps);
        });

        // Number inputs with validation
        this.elements.numSpeakersInput.addEventListener('input', UIUtils.debounce(() => {
            const value = parseInt(this.elements.numSpeakersInput.value);
            this.settings.numSpeakers = (value >= 1 && value <= 20) ? value : null;
            this.saveSettings();
            this.emitSettingsChange();
        }, 500));

        this.elements.chunkDurationInput.addEventListener('input', UIUtils.debounce(() => {
            const value = parseFloat(this.elements.chunkDurationInput.value);
            this.settings.chunkDurationSec = (value >= 10 && value <= 300) ? value : null;
            this.saveSettings();
            this.emitSettingsChange();
        }, 500));

        this.elements.overlapDurationInput.addEventListener('input', UIUtils.debounce(() => {
            const value = parseFloat(this.elements.overlapDurationInput.value);
            this.settings.overlapDurationSec = (value >= 0 && value <= 30) ? value : null;
            this.saveSettings();
            this.emitSettingsChange();
        }, 500));
    }

    // Get current settings for API calls
    getSettings() {
        return {
            returnTimestamps: true, // ALWAYS request timestamps regardless of UI toggle
            numSpeakers: this.settings.numSpeakers,
            chunkDurationSec: this.settings.chunkDurationSec,
            overlapDurationSec: this.settings.overlapDurationSec
        };
    }

    // Get UI preferences
    getUIPreferences() {
        return {
            showSegments: this.settings.showSegments,
            showWordTimestamps: this.settings.showWordTimestamps
        };
    }

    // Load settings from localStorage
    loadSettings() {
        const saved = UIUtils.loadFromLocalStorage('parakeet-settings', {});

        // Merge saved settings, but ensure sane defaults
        this.settings = {
            returnTimestamps: true, // Always true for API
            showSegments: saved.showSegments !== undefined ? saved.showSegments : true,
            showWordTimestamps: saved.showWordTimestamps !== undefined ? saved.showWordTimestamps : true,
            numSpeakers: saved.numSpeakers || null,
            chunkDurationSec: saved.chunkDurationSec || null,
            overlapDurationSec: saved.overlapDurationSec || null
        };

        // Update UI to match loaded settings
        this.elements.segmentsToggle.checked = this.settings.showSegments;
        this.elements.timestampsToggle.checked = this.settings.showWordTimestamps;

        if (this.settings.numSpeakers) {
            this.elements.numSpeakersInput.value = this.settings.numSpeakers;
        }

        if (this.settings.chunkDurationSec) {
            this.elements.chunkDurationInput.value = this.settings.chunkDurationSec;
        }

        if (this.settings.overlapDurationSec) {
            this.elements.overlapDurationInput.value = this.settings.overlapDurationSec;
        }

        console.log('DEBUG: Settings loaded and applied:', this.settings);
    }

    // Save settings to localStorage
    saveSettings() {
        UIUtils.saveToLocalStorage('parakeet-settings', this.settings);
        console.log('DEBUG: Settings saved:', this.settings);
    }

    // Emit settings change event
    emitSettingsChange() {
        const preferences = this.getUIPreferences();
        const event = new CustomEvent('settingsChanged', {
            detail: {
                settings: this.getSettings(),
                preferences: preferences
            }
        });
        document.dispatchEvent(event);
        console.log('DEBUG: Settings change event emitted:', preferences);
    }

    // Reset to defaults
    resetToDefaults() {
        this.settings = {
            returnTimestamps: true,
            showSegments: true,
            showWordTimestamps: true,
            numSpeakers: null,
            chunkDurationSec: null,
            overlapDurationSec: null
        };

        // Update UI
        this.elements.segmentsToggle.checked = true;
        this.elements.timestampsToggle.checked = true;
        this.elements.numSpeakersInput.value = '';
        this.elements.chunkDurationInput.value = '';
        this.elements.overlapDurationInput.value = '';

        this.saveSettings();
        this.emitSettingsChange();

        UIUtils.showToast('Settings reset to defaults', 'info');
    }

    // Get settings summary for display
    getSettingsSummary() {
        const summary = [];

        if (this.settings.numSpeakers) {
            summary.push(`${this.settings.numSpeakers} speakers`);
        } else {
            summary.push('Auto-detect speakers');
        }

        if (this.settings.returnTimestamps) {
            summary.push('With timestamps');
        }

        if (this.settings.chunkDurationSec) {
            summary.push(`${this.settings.chunkDurationSec}s chunks`);
        }

        return summary.join(', ');
    }
}
