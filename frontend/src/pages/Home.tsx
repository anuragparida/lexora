import { useState, useEffect } from 'react'
import '../index.css'

interface Example {
  id: number
  german: string
  english: string
}

interface VerbConjugation {
  id: number
  infinitive: string
  present_3rd_person: string | null
  simple_past: string | null
  participle: string | null
}

interface Word {
  id: number
  word: string
  word_type: string | null
  frequency: string | null
  level: string | null
  translations: string | null
  conjugation: string | null
  additional_info: string | null
  examples: Example[]
  is_complete: boolean
  verb_conjugation: VerbConjugation | null
}

interface WordListResponse {
  items: Word[]
  total: number
  page: number
  page_size: number
}

interface FilterOptions {
  word_types: string[]
  frequencies: string[]
}

interface DeckInfo {
  filename: string
  created: number
  size: number
}

type CardDirection = 'both' | 'de-en' | 'en-de'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function WordList({ words }: { words: Word[] }) {
  return (
    <div className="grid gap-4">
      {words.map((word) => (
        <div
          key={word.id}
          className="rounded-lg shadow-sm border p-6 hover:shadow-md transition-shadow bg-slate-800 border-slate-700"
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-semibold text-slate-100">
                  {word.word}
                </h2>
              </div>
              <div className="flex gap-2 mt-1">
                {word.word_type && (
                  <span className="px-2 py-1 bg-blue-900 text-blue-200 text-xs rounded-full font-medium">
                    {word.word_type}
                  </span>
                )}
                {word.level && (
                  <span className="px-2 py-1 bg-green-900 text-green-200 text-xs rounded-full font-medium">
                    {word.level}
                  </span>
                )}
                {word.frequency && (
                  <span className="px-2 py-1 bg-purple-900 text-purple-200 text-xs rounded-full font-medium">
                    Freq: {word.frequency}
                  </span>
                )}
              </div>
            </div>
          </div>

          {word.translations && (
            <p className="text-slate-300 mb-3">
              <span className="font-medium">Translation:</span>{' '}
              {word.translations}
            </p>
          )}

          {word.verb_conjugation && (
            <div className="mt-3 mb-4 bg-amber-900/30 border border-amber-700 rounded-lg p-4">
              <p className="text-sm font-medium text-amber-200 uppercase tracking-wide mb-2">
                Conjugation
              </p>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-amber-400 text-xs mb-1">3rd Person Present</p>
                  <p className="text-slate-100 font-medium">{word.verb_conjugation.present_3rd_person || '-'}</p>
                </div>
                <div>
                  <p className="text-amber-400 text-xs mb-1">Simple Past</p>
                  <p className="text-slate-100 font-medium">{word.verb_conjugation.simple_past || '-'}</p>
                </div>
                <div>
                  <p className="text-amber-400 text-xs mb-1">Participle</p>
                  <p className="text-slate-100 font-medium">{word.verb_conjugation.participle || '-'}</p>
                </div>
              </div>
            </div>
          )}

          {word.examples.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-sm font-medium text-slate-500 uppercase tracking-wide">
                Examples
              </p>
              {word.examples.slice(0, 2).map((example) => (
                <div
                  key={example.id}
                  className="bg-slate-900 rounded-md p-3"
                >
                  <p className="text-slate-100 font-medium">
                    {example.german}
                  </p>
                  <p className="text-slate-400 text-sm mt-1">
                    {example.english}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function WordTypeCheckboxes({
  options,
  selected,
  onChange,
}: {
  options: string[]
  selected: string[]
  onChange: (selected: string[]) => void
}) {
  const toggleOption = (option: string) => {
    if (selected.includes(option)) {
      onChange(selected.filter((s) => s !== option))
    } else {
      onChange([...selected, option])
    }
  }

  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-slate-400 mb-2">
        Word Types
      </label>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => (
          <label
            key={option}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm cursor-pointer transition-colors ${
              selected.includes(option)
                ? 'bg-blue-600 text-white'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <input
              type="checkbox"
              checked={selected.includes(option)}
              onChange={() => toggleOption(option)}
              className="hidden"
            />
            <span>{option}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function FrequencyCheckboxes({
  options,
  selected,
  onChange,
}: {
  options: string[]
  selected: string[]
  onChange: (selected: string[]) => void
}) {
  const toggleOption = (option: string) => {
    if (selected.includes(option)) {
      onChange(selected.filter((s) => s !== option))
    } else {
      onChange([...selected, option])
    }
  }

  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-slate-400 mb-2">
        Frequency Levels
      </label>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => (
          <label
            key={option}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm cursor-pointer transition-colors ${
              selected.includes(option)
                ? 'bg-purple-600 text-white'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <input
              type="checkbox"
              checked={selected.includes(option)}
              onChange={() => toggleOption(option)}
              className="hidden"
            />
            <span>Level {option}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function App() {
  const [words, setWords] = useState<Word[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedWordTypes, setSelectedWordTypes] = useState<string[]>([])
  const [selectedFrequencies, setSelectedFrequencies] = useState<string[]>([])
  const [cardDirection, setCardDirection] = useState<CardDirection>('both')
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({ word_types: [], frequencies: [] })
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [generatingDeck, setGeneratingDeck] = useState(false)
  const [decks, setDecks] = useState<DeckInfo[]>([])
  const [showDecks, setShowDecks] = useState(false)
  const pageSize = 20

  useEffect(() => {
    fetchFilterOptions()
    fetchWords()
    fetchDecks()
  }, [page, selectedWordTypes, selectedFrequencies])

  const fetchFilterOptions = async () => {
    try {
      const response = await fetch(`${API_URL}/words/filters/options`)
      if (response.ok) {
        const data = await response.json()
        setFilterOptions(data)
      }
    } catch (err) {
      console.error('Failed to fetch filter options:', err)
    }
  }

  const fetchWords = async () => {
    try {
      setLoading(true)
      const skip = (page - 1) * pageSize
      let url = searchQuery
        ? `${API_URL}/words/search?q=${encodeURIComponent(searchQuery)}&skip=${skip}&limit=${pageSize}`
        : `${API_URL}/words?skip=${skip}&limit=${pageSize}`

      if (selectedWordTypes.length > 0) {
        selectedWordTypes.forEach((wt) => {
          url += `&word_types=${encodeURIComponent(wt)}`
        })
      }
      if (selectedFrequencies.length > 0) {
        selectedFrequencies.forEach((freq) => {
          url += `&frequencies=${encodeURIComponent(freq)}`
        })
      }

      const response = await fetch(url)
      if (!response.ok) throw new Error('Failed to fetch words')

      const data: WordListResponse = await response.json()
      setWords(data.items)
      setTotal(data.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const fetchDecks = async () => {
    try {
      const response = await fetch(`${API_URL}/decks/list`)
      if (response.ok) {
        const data = await response.json()
        setDecks(data.decks)
      }
    } catch (err) {
      console.error('Failed to fetch decks:', err)
    }
  }

  const generateDeck = async () => {
    try {
      setGeneratingDeck(true)
      let url = `${API_URL}/decks/generate?direction=${cardDirection}`

      if (selectedWordTypes.length > 0) {
        selectedWordTypes.forEach((wt) => {
          url += `&word_types=${encodeURIComponent(wt)}`
        })
      }
      if (selectedFrequencies.length > 0) {
        selectedFrequencies.forEach((freq) => {
          url += `&frequencies=${encodeURIComponent(freq)}`
        })
      }

      const response = await fetch(url, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to generate deck')

      const data = await response.json()
      alert(`Deck generated: ${data.filename}`)
      fetchDecks()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to generate deck')
    } finally {
      setGeneratingDeck(false)
    }
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    setPage(1)
    fetchWords()
  }

  const clearFilters = () => {
    setSearchQuery('')
    setSelectedWordTypes([])
    setSelectedFrequencies([])
    setCardDirection('both')
    setPage(1)
  }

  const totalPages = Math.ceil(total / pageSize)

  return (
    <div className="h-screen bg-slate-950 flex overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`bg-slate-900 border-r border-slate-700 flex-shrink-0 transition-all duration-300 h-full overflow-y-auto ${
          sidebarOpen ? 'w-80' : 'w-0 overflow-hidden'
        }`}
      >
        <div className="p-6">
          <h2 className="text-lg font-semibold text-slate-100 mb-6">Filters</h2>

          {/* Search */}
          <form onSubmit={handleSearch} className="mb-6">
            <label className="block text-sm font-medium text-slate-400 mb-2">
              Search
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search words..."
                className="flex-1 px-3 py-2 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm bg-slate-800 text-slate-100"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
              >
                Go
              </button>
            </div>
          </form>

          {/* Word Type Filter */}
          <WordTypeCheckboxes
            options={filterOptions.word_types}
            selected={selectedWordTypes}
            onChange={(selected) => {
              setSelectedWordTypes(selected)
              setPage(1)
            }}
          />

          {/* Frequency Filter */}
          <FrequencyCheckboxes
            options={[...filterOptions.frequencies].sort((a, b) => Number(b) - Number(a))}
            selected={selectedFrequencies}
            onChange={(selected) => {
              setSelectedFrequencies(selected)
              setPage(1)
            }}
          />

          {/* Card Direction */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-400 mb-2">
              Card Direction
            </label>
            <select
              value={cardDirection}
              onChange={(e) => setCardDirection(e.target.value as CardDirection)}
              className="w-full px-3 py-2 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm bg-slate-800 text-slate-100"
            >
              <option value="both">Both Directions</option>
              <option value="de-en">German → English</option>
              <option value="en-de">English → German</option>
            </select>
          </div>

          {/* Deck Builder */}
          <div className="mb-6 p-4 bg-blue-900/30 rounded-lg border border-blue-700">
            <h3 className="text-sm font-semibold text-blue-200 mb-3">Anki Deck Builder</h3>

            <button
              onClick={generateDeck}
              disabled={generatingDeck}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {generatingDeck ? 'Generating...' : 'Generate Deck'}
            </button>
          </div>

          {/* Generated Decks */}
          {decks.length > 0 && (
            <div className="mb-6">
              <button
                onClick={() => setShowDecks(!showDecks)}
                className="flex items-center justify-between w-full text-sm font-medium text-slate-300 hover:text-slate-100 transition-colors mb-2"
              >
                <span>Generated Decks ({decks.length})</span>
                <svg
                  className={`w-4 h-4 transition-transform ${showDecks ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showDecks && (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {decks.map((deck) => (
                    <div
                      key={deck.filename}
                      className="flex items-center justify-between p-2 bg-slate-800 rounded text-xs"
                    >
                      <span className="text-slate-300 truncate flex-1 mr-2">{deck.filename}</span>
                      <span className="text-slate-500 whitespace-nowrap">
                        {(deck.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Clear Filters */}
          <button
            onClick={clearFilters}
            className="w-full px-4 py-2 border border-slate-600 text-slate-300 rounded-lg hover:bg-slate-800 transition-colors text-sm font-medium"
          >
            Clear All Filters
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Header */}
        <header className="bg-slate-900 shadow-sm border-b border-slate-700 flex-shrink-0">
          <div className="px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                aria-label="Toggle sidebar"
              >
                <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <div>
                <h1 className="text-2xl font-bold text-slate-100">German Vocabulary</h1>
                <p className="text-sm text-slate-400">{total.toLocaleString()} words</p>
              </div>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 p-6 overflow-y-auto">
          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              <p className="mt-2 text-slate-400">Loading words...</p>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <p className="text-red-400">{error}</p>
            </div>
          ) : (
            <>
              <WordList words={words} />

              {totalPages > 1 && (
                <div className="flex justify-center items-center gap-2 mt-8">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="px-4 py-2 border border-slate-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800 transition-colors text-slate-300"
                  >
                    Previous
                  </button>
                  <span className="text-slate-400">
                    Page {page} of {totalPages}
                  </span>
                  <button
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    className="px-4 py-2 border border-slate-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800 transition-colors text-slate-300"
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export { App as Home }
