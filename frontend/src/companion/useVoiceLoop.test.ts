import { describe, expect, it } from 'vitest'
import { replyLangTag } from './useVoiceLoop'

describe('replyLangTag', () => {
  it('picks the voice locale from the script used in the reply', () => {
    expect(replyLangTag('ನಾನು ಚೆನ್ನಾಗಿದ್ದೀನಿ', 'en-IN')).toBe('kn-IN')
    expect(replyLangTag('நான் நல்லா இருக்கேன்', 'en-IN')).toBe('ta-IN')
    expect(replyLangTag('സുഖമാണ്', 'en-IN')).toBe('ml-IN')
    expect(replyLangTag('मैं ठीक हूँ', 'en-IN')).toBe('hi-IN')
  })
  it('uses an English voice for Romanized replies', () => {
    expect(replyLangTag('Chennagiddini maga!', 'kn-IN')).toBe('en-IN')
    expect(replyLangTag('Doing well!', 'en-US')).toBe('en-US')
  })
})
