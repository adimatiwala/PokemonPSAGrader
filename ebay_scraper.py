from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, requests, os, sys

def scrape_ebay_psa_images(
    query="Pokemon Cards PSA",
    grade="10",
    output_dir="downloads",
    headless=False
):
    """
    Applies parameters  and extracts the first 2 pages of ebay for Pokemon card images 
    """
    # Stores all image URLs 
    urls = []

    # 1) Build the search URL
    query = query.replace(" ", "+") + "+" + f"{grade}"
    
    # 2) Iterate through Pages 1 - 2
    for i in range(1,3):
        base_url = (
            f"https://www.ebay.com/sch/i.html?_dcat=183454&_fsrp=1&rt=nc&_from=R40&_nkw={query}&_sacat=0&Grade={grade}&_ipg=240&_pgn={i}"
        )
        
        # 3) Open Ebay
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.set_window_size(1200, 800)
        driver.get(base_url)
        
        # 4) Remove “Picked for you” sections (if any)
        driver.execute_script("""
            document.querySelectorAll('h2').forEach(h => {
                if (h.innerText.trim().startsWith('Picked for you')) {
                    const sec = h.closest('section') || h.parentElement;
                    if (sec) sec.remove();
                }
            });
        """)
        
        # 5) Wait until at least one listing image is present
        wait = WebDriverWait(driver, 10)
        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "div.s-item__image-wrapper.image-treatment img"
            )))
        except:
            print("No images found. Check selector or network.")
            driver.quit()
            sys.exit(1)
        
        # 6) Collect image URLs
        elems = driver.find_elements(
            By.CSS_SELECTOR,
            "div.s-item__image-wrapper.image-treatment img"
        )
        print(f"Found {len(elems)} <img> elements")

        for img in elems:
            src = img.get_attribute("src") or ""
            alt = img.get_attribute("alt") or ""
            # Only keeping the ones that mention PSA and desired grade
            if "/s-l500" in src and f"PSA {grade}" in alt:
                urls.append(src)
        print(f"Filtered down to {len(urls)} PSA-{grade} images")
    
    # 7) Downloading all the images
    os.makedirs(output_dir, exist_ok=True)
    for i, url in enumerate(urls, start=1):
        try:
            print(f"Downloading ({i}/{len(urls)}): {url}")
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            filename = os.path.join(output_dir, f"card_{i}.jpg")
            with open(filename, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"  ✖ Failed to download {url}: {e}")
    
    driver.quit()
    print(f"Done. Downloaded {len(urls)} images to '{output_dir}'")

if __name__ == "__main__":
    for grade in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        scrape_ebay_psa_images(
            query="Pokemon Cards PSA",
            grade=grade,
            output_dir=f"psa{grade}_images",
            headless=False
        )
