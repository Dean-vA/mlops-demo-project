// Settings Component - Handles user preferences and transcription settings
class SettingsComponent {
    constructor(state) {
        this.state = state;
        this.container = document.getElementById('settings-component');
        this.settings = {
            returnTimestamps: true,
            showSegments: true,
            showWordTimestamps: true,
            numSpeakers: null
        };
    }

    async init() {
        this.render();
        this.setupEventListeners();
        this.loadSettings();
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
        });

        this.elements.timestampsToggle.addEventListener('change', () => {
            this.settings.showWordTimestamps = this.elements.timestampsToggle.checked;
            this.settings.returnTimestamps = this.elements.timestampsToggle.checked;
            this.saveSettings();
            this.emitSettingsChange();
        });

        // Number inputs with validation
        this.elements.numSpeakersInput.addEventListener('input', UIUtils.debounce(() => {
            const value = parseInt(this.elements.numSpeakersInput.value);
            this.settings.numSpeakers = (value >= 1 && value <= 20) ? value : null;
            this.saveSettings();
        }, 500));

        this.elements.chunkDurationInput.addEventListener('input', UIUtils.debounce(() => {
            const value = parseFloat(this.elements.chunkDurationInput.value);
            this.settings.chunkDurationSec = (value >= 10 && value <= 300) ? value : null;
            this.saveSettings();
        }, 500));

        this.elements.overlapDurationInput.addEventListener('input', UIUtils.debounce(() => {
            const value = parseFloat(this.elements.overlapDurationInput.value);
            this.settings.overlapDurationSec = (value >= 0 && value <= 30) ? value : null;
            this.saveSettings();
        }, 500));

        // Listen for results to show/hide sections
        document.addEventListener('transcriptionComplete', (event) => {
            this.updateVisibilityBasedOnResults(event.detail);
        });
    }

    // Get current settings for API calls
    getSettings() {
        return {
            returnTimestamps: this.settings.returnTimestamps,
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

    // Update visibility of result sections based on settings
    updateVisibilityBasedOnResults(data) {
        const preferences = this.getUIPreferences();

        // Show/hide segments section
        const segmentsSection = document.getElementById('segments-section');
        if (segmentsSection) {
            const hasSegments = data.transcription?.segments?.length > 0 ||
                               data.segments?.length > 0;
            segmentsSection.style.display =
                (preferences.showSegments && hasSegments) ? 'block' : 'none';
        }

        // Show/hide timestamps section
        const timestampsSection = document.getElementById('timestamps-section');
        if (timestampsSection) {
            const hasTimestamps = data.transcription?.timestamps?.word?.length > 0 ||
                                 data.timestamps?.word?.length > 0;
            timestampsSection.style.display =
                (preferences.showWordTimestamps && hasTimestamps) ? 'block' : 'none';
        }
    }

    // Validate settings
    validateSettings() {
        const errors = [];

        if (this.settings.numSpeakers !== null) {
            if (this.settings.numSpeakers < 1 || this.settings.numSpeakers > 20) {
                errors.push('Number of speakers must be between 1 and 20');
            }
        }

        if (this.settings.chunkDurationSec !== null) {
            if (this.settings.chunkDurationSec < 10 || this.settings.chunkDurationSec > 300) {
                errors.push('Chunk duration must be between 10 and 300 seconds');
            }
        }

        if (this.settings.overlapDurationSec !== null) {
            if (this.settings.overlapDurationSec < 0 || this.settings.overlapDurationSec > 30) {
                errors.push('Overlap duration must be between 0 and 30 seconds');
            }
        }

        return errors;
    }

    // Load settings from localStorage
    loadSettings() {
        const saved = UIUtils.loadFromLocalStorage('parakeet-settings', {});
        this.settings = { ...this.settings, ...saved };

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
    }

    // Save settings to localStorage
    saveSettings() {
        UIUtils.saveToLocalStorage('parakeet-settings', this.settings);
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

    // Export settings
    exportSettings() {
        const settingsJson = JSON.stringify(this.settings, null, 2);
        const blob = new Blob([settingsJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = 'parakeet-settings.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        UIUtils.showToast('Settings exported successfully', 'success');
    }

    // Import settings
    async importSettings(file) {
        try {
            const text = await file.text();
            const importedSettings = JSON.parse(text);

            // Validate imported settings
            const validKeys = Object.keys(this.settings);
            const filteredSettings = {};

            for (const key of validKeys) {
                if (key in importedSettings) {
                    filteredSettings[key] = importedSettings[key];
                }
            }

            this.settings = { ...this.settings, ...filteredSettings };
            this.loadSettings(); // Update UI
            this.saveSettings();
            this.emitSettingsChange();

            UIUtils.showToast('Settings imported successfully', 'success');
        } catch (error) {
            console.error('Error importing settings:', error);
            UIUtils.showToast('Failed to import settings', 'error');
        }
    }

    // Emit settings change event
    emitSettingsChange() {
        const event = new CustomEvent('settingsChanged', {
            detail: {
                settings: this.getSettings(),
                preferences: this.getUIPreferences()
            }
        });
        document.dispatchEvent(event);
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
