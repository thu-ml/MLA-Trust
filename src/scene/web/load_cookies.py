from playwright.sync_api import sync_playwright
import json

def run(playwright):
    browser = playwright.chromium.launch(headless=False) 
    context = browser.new_context()

    try:
        with open('taobao.json', 'r') as f: # you should change the file name to the cookies file you want to load
            cookies = json.load(f)
            context.add_cookies(cookies)
    except FileNotFoundError:
        print("Cookies file not found. Will proceed with new login.")

    page = context.new_page()
    page.goto('https://www.amazon.com/') # you should change the url to the website you want to visit

    coupon_button = page.locator('button:text("Add to cart")')
    print(coupon_button)
    if coupon_button.count() > 0:
        coupon_button.first.click()
        print("Add to cart button clicked")
    else:
        print("Add to cart button not found")

    # Add any necessary web interactions or checks here

    # Wait for user input to prevent the browser from closing immediately after the script ends
    input("Press Enter to close browser...")

    # Save cookies to file if needed
    # with open('taobao.json', 'w') as f:
    #     json.dump(context.cookies(), f)

    context.close()
    browser.close()

with sync_playwright() as playwright:
    run(playwright)