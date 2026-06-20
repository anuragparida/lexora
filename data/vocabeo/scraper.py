import asyncio
import json
import time
from playwright.async_api import async_playwright
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class WordEntry:
    word: str
    word_type: str
    frequency: Optional[str]
    level: Optional[str]
    translations: List[str]
    conjugation: Optional[str]
    examples: List[dict]
    additional_info: dict


class VocabeoScraper:
    def __init__(
        self, max_duration_seconds=None, click_delay_ms=50, scroll_delay_ms=100
    ):
        self.base_url = "https://vocabeo.com/browse"
        self.words_data = []
        self.seen_words = set()
        self.max_duration = max_duration_seconds
        self.start_time = None
        self.click_delay = click_delay_ms
        self.scroll_delay = scroll_delay_ms

    async def scrape(self):
        self.start_time = time.time()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()

            print("Navigating to vocabeo.com/browse...")
            await page.goto(self.base_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            print("Starting to scrape words...")
            print("Press Ctrl+C to stop at any time and save progress.")

            try:
                await self._scrape_all_words(page)
            except KeyboardInterrupt:
                print("\nScraping interrupted by user.")
            finally:
                self._save_data()
                await browser.close()

    def _should_stop(self):
        """Check if we should stop based on time limit."""
        if self.max_duration and self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_duration:
                print(f"\nReached time limit of {self.max_duration} seconds.")
                return True
        return False

    async def _scrape_all_words(self, page):
        """Scrape all words using keyboard navigation (arrow down)."""

        word_count = 0
        max_words = 6260
        consecutive_no_new = 0
        max_consecutive_no_new = 10
        last_word = None

        # Focus the list by clicking the first item
        print("Focusing the word list...")
        first_item = await page.query_selector(".virtual-list-inner > div")
        if first_item:
            await first_item.click()
            await page.wait_for_timeout(200)
        else:
            print("ERROR: Could not find first item in list!")
            return

        while word_count < max_words:
            if self._should_stop():
                break

            try:
                # Scrape current word details
                word_data = await self._scrape_current_word_details(page)

                if word_data:
                    if word_data.word not in self.seen_words:
                        self.words_data.append(asdict(word_data))
                        self.seen_words.add(word_data.word)
                        word_count += 1
                        consecutive_no_new = 0
                        last_word = word_data.word

                        if word_count % 50 == 0:
                            elapsed = time.time() - self.start_time
                            rate = word_count / elapsed if elapsed > 0 else 0
                            print(
                                f"Scraped {word_count} words... "
                                f"(rate: {rate:.1f} words/sec, current: {word_data.word})"
                            )
                            self._save_data(f"vocabeo_progress_{word_count}.json")
                    else:
                        # Word already seen - we might be looping
                        consecutive_no_new += 1
                        if consecutive_no_new >= max_consecutive_no_new:
                            print(
                                f"\nNo new words for {consecutive_no_new} attempts. "
                                f"Total scraped: {word_count}"
                            )
                            break
                else:
                    consecutive_no_new += 1

                # Press arrow down to go to next word
                await page.keyboard.press("ArrowDown")
                await page.wait_for_timeout(self.click_delay)

                # Check if we're stuck (same word after pressing down)
                current_word_check = await page.evaluate("""() => {
                    const transDiv = document.querySelector('#translation, [id*="translation"]');
                    if (transDiv) {
                        const deuDiv = transDiv.querySelector('#deu, [id*="deu"]');
                        return deuDiv ? deuDiv.textContent.trim() : null;
                    }
                    return null;
                }""")

                if current_word_check == last_word:
                    # We're at the bottom of the visible list, need to scroll
                    # Scroll the virtual list wrapper to bring more items into view
                    await page.evaluate("""() => {
                        const wrapper = document.querySelector('#virtual-list-wrapper');
                        if (wrapper) {
                            // Scroll up by a few items to create buffer, then we can continue going down
                            wrapper.scrollTop += 30.4 * 5; // scroll up by ~5 items
                        }
                    }""")
                    await page.wait_for_timeout(self.scroll_delay)

                    # Try arrow down again after scroll
                    await page.keyboard.press("ArrowDown")
                    await page.wait_for_timeout(self.click_delay)

            except Exception as e:
                print(f"Error in main loop: {e}")
                # Try to recover by pressing down
                await page.keyboard.press("ArrowDown")
                await page.wait_for_timeout(self.click_delay)
                continue

        elapsed = time.time() - self.start_time
        rate = word_count / elapsed if elapsed > 0 else 0
        print(f"\nScraping complete! Total words scraped: {word_count}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Average rate: {rate:.1f} words/sec")

    async def _scrape_current_word_details(self, page) -> Optional[WordEntry]:
        """Scrape the details from the right side panel."""

        try:
            # Extract all data in a single evaluate call for speed
            data = await page.evaluate("""() => {
                const result = {
                    word: null,
                    word_type: null,
                    frequency: null,
                    level: null,
                    translations: [],
                    conjugation: null,
                    examples: [],
                    additional_info: {}
                };
                
                // Extract word
                const transDiv = document.querySelector('#translation, [id*="translation"]');
                if (transDiv) {
                    const deuDiv = transDiv.querySelector('#deu, [id*="deu"]');
                    if (deuDiv) {
                        result.word = deuDiv.textContent.trim();
                    }
                }
                
                if (!result.word) return null;
                
                // Extract word type, frequency, level from top section
                const top = document.querySelector('.top');
                if (top) {
                    const text = top.textContent;
                    
                    // Word type - first meaningful div
                    const divs = top.querySelectorAll('div');
                    for (let div of divs) {
                        const divText = div.textContent.trim();
                        if (divText && divText.length > 0 && 
                            !divText.includes('Frequency') && 
                            !divText.includes('Level') && 
                            !divText.includes('plural') &&
                            !divText.includes('words saved')) {
                            result.word_type = divText;
                            break;
                        }
                    }
                    
                    // Frequency
                    const freqMatch = text.match(/Frequency\s*(\d+)/);
                    if (freqMatch) result.frequency = freqMatch[1];
                    
                    // Level
                    const levelMatch = text.match(/Level\s*(A\d|B\d|C\d)/);
                    if (levelMatch) result.level = levelMatch[1];
                    
                    // Plural
                    if (text.includes('plural')) {
                        const pluralDiv = top.querySelector('div:last-child');
                        if (pluralDiv) {
                            result.additional_info.plural = pluralDiv.textContent.trim();
                        }
                    }
                    
                    // Gender
                    const genderMatch = text.match(/(masculine|feminine|neuter)/);
                    if (genderMatch) result.additional_info.gender = genderMatch[1];
                }
                
                // Extract translations
                if (transDiv) {
                    const enDiv = transDiv.querySelector('#en, [id*="en"]');
                    if (enDiv) {
                        const buttons = enDiv.querySelectorAll('button');
                        result.translations = Array.from(buttons)
                            .map(b => b.textContent.trim())
                            .filter(t => t.length > 0);
                    }
                }
                
                // Extract conjugation
                const conjElements = document.querySelectorAll('[class*="conjug"], [class*="verb-form"]');
                for (let el of conjElements) {
                    const text = el.textContent.trim();
                    if (text && text.length > 0) {
                        result.conjugation = text;
                        break;
                    }
                }
                
                // Extract examples - get text from parts-and-translation div
                const sentences = document.querySelectorAll('.sentence, [class*="sentence"]');
                for (let s of sentences) {
                    const partsAndTrans = s.querySelector('.parts-and-translation');
                    if (partsAndTrans) {
                        // Get all text nodes and separate by structure
                        const partsDiv = partsAndTrans.querySelector('.sentence-parts');
                        const transDiv = partsAndTrans.querySelector('.sentence-translation');
                        
                        if (partsDiv && transDiv) {
                            const german = partsDiv.textContent.trim();
                            const english = transDiv.textContent.trim();
                            
                            if (german || english) {
                                result.examples.push({
                                    german: german,
                                    english: english
                                });
                            }
                        } else {
                            // Fallback: try to split by looking for German/English patterns
                            const fullText = partsAndTrans.textContent.trim();
                            // German sentences typically end with period before English
                            const match = fullText.match(/^(.+?\.\s*)([A-Z].+)$/);
                            if (match) {
                                result.examples.push({
                                    german: match[1].trim(),
                                    english: match[2].trim()
                                });
                            } else {
                                result.examples.push({
                                    german: fullText,
                                    english: ''
                                });
                            }
                        }
                    }
                }
                
                return result;
            }""")

            if not data:
                return None

            return WordEntry(
                word=data["word"],
                word_type=data.get("word_type") or "Unknown",
                frequency=data.get("frequency"),
                level=data.get("level"),
                translations=data.get("translations", []),
                conjugation=data.get("conjugation"),
                examples=data.get("examples", []),
                additional_info=data.get("additional_info", {}),
            )

        except Exception as e:
            return None

    def _save_data(self, filename: str = "vocabeo_words.json"):
        """Save scraped data to JSON file."""
        filepath = f"{filename}"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.words_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.words_data)} words to {filepath}")


async def main():
    # Fast settings for keyboard navigation approach
    scraper = VocabeoScraper(
        max_duration_seconds=None, click_delay_ms=50, scroll_delay_ms=100
    )

    await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
