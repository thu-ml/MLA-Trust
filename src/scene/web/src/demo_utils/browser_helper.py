# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import random
import re
import socket
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import toml
from playwright.async_api import Error as PlaywrightError
from playwright.sync_api import Playwright

list_us_cities = [
    ["New York", 40.77, -73.98],
    ["Log Angeles", 34.05, -118.24],
    ["Chicago", 41.88, -87.63],
    ["Houston", 29.76, -95.36],
    ["Phoneix", 33.45, -112.07],
    ["Philadelphia", 39.95, -75.17],
    ["San Antonio", 29.53, -98.47],
    ["San Diego", 32.78, -117.15],
    ["Dallas", 32.79, -96.80],
    ["San Jose", 37.30, -121.87],
    ["Austin", 30.27, -97.74],
    ["Jacksonville", 30.32, -81.66],
    ["Fort Worth", 32.75, -97.33],
    ["Columbus", 39.98, -82.98],
    ["Indianapolis", 39.77, -86.15],
    ["Charlotte", 35.23, -80.84],
    ["San Francisco", 37.77, -122.42],
    ["Seattle", 47.61, -122.33],
    ["Denver", 39.74, -104.98],
    ["Oklahoma City", 35.48, -97.53],
    ["Nashville", 36.16, -86.78],
    ["El Paso", 31.79, -106.42],
    ["Washington", 38.91, -77.01],
    ["Boston", 42.36, -71.06],
    ["Las Vegas", 36.19, -115.22],
    ["Portland", 45.52, -122.68],
    ["Detroit", 42.33, -83.05],
    ["Louisville", 38.25, -85.76],
    ["Memphis", 35.15, -90.05],
    ["Balitmore", 39.29, -76.61],
    ["Anchorage", 61.22, -149.90],
    ["Phoenix", 33.45, -112.07],
    ["Little Rock", 34.74, -92.33],
    ["Bridgeport", 41.18, -73.19],
    ["Wilmington", 39.75, -75.54],
    ["Atlanta", 33.75, -84.39],
    ["Honolulu", 21.31, -157.86],
    ["New Orleans", 29.95, -90.07],
    ["Minneapolis", 44.98, -93.27],
    ["Jackson", 32.30, -90.18],
    ["Kansas City", 39.10, -94.58],
    ["Newark", 40.73, -74.17],
    ["Salt Lake City", 40.75, -111.89],
    ["Milwaukee", 43.04, -87.96],
]

ignore_args = [
    "--enable-automation",
    "--disable-field-trial-config",
    "--disable-background-networking",
    "--enable-features=" + ("NetworkService,NetworkServiceInProcess"),
    "--disable-features="
    + (
        "ImprovedCookieControls"
        ",LazyFrameLoading"
        ",GlobalMediaControls"
        ",DestroyProfileOnBrowserClose"
        ",MediaRouter"
        ",DialMediaRouteProvider"
        ",AcceptCHFrame"
        ",AutoExpandDetailsElement"
        ",CertificateTransparencyComponentUpdater"
        ",AvoidUnnecessaryBeforeUnloadCheckSync"
        ",Translate"
    ),
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-back-forward-cache",
    "--disable-breakpad",
    "--disable-client-side-phishing-detection",
    "--disable-component-extensions-with-background-pages",
    "--disable-component-update",
    "--no-default-browser-check",
    "--disable-default-apps",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-features",
    "--allow-pre-commit-input",
    "--disable-hang-monitor",
    "--disable-ipc-flooding-protection",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-renderer-backgrounding",
    "--disable-sync",
    "--force-color-profile",
    "--metrics-recording-only",
    "--no-first-run",
    "--password-store",
    "--use-mock-keychain",
    "--no-service-autorun",
    "--export-tagged-pdf",
    "--no_sandbox",
]


async def normal_launch_async(playwright: Playwright):
    browser = await playwright.chromium.launch(
        # traces_dir=None,
        # proxy={"server": "localhost:3690"},
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
        ],
        # ignore_default_args=ignore_args,
        # chromium_sandbox=False,
    )
    return browser


def normal_launch(playwright: Playwright):
    browser = playwright.chromium.launch(
        headless=False,
        args=[
            "--incognito",
            "--disable-blink-features=AutomationControlled",
        ],
        ignore_default_args=ignore_args,
    )
    return browser


async def normal_new_context_async(
    browser,
    storage_state=None,
    har_path=None,
    video_path=None,
    tracing=False,
    trace_screenshots=False,
    trace_snapshots=False,
    trace_sources=False,
    locale=None,
    geolocation=None,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    viewport: dict = {"width": 1280, "height": 720},
):
    city = random.choice(list_us_cities)
    context = await browser.new_context(
        storage_state=storage_state,
        user_agent=user_agent,
        viewport=viewport,
        locale=locale,
        record_har_path=har_path,
        record_video_dir=video_path,
        geolocation=geolocation,
    )

    if tracing:
        await context.tracing.start(
            screenshots=trace_screenshots,
            snapshots=trace_snapshots,
            sources=trace_sources,
        )
    return context


def normal_new_context(
    browser,
    storage_state=None,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    viewport: dict = {"width": 1280, "height": 720},
):
    return browser.new_context(
        storage_state=storage_state,
        user_agent=user_agent,
        viewport=viewport,
    )


def persistent_launch(playwright: Playwright, user_data_dir: str = ""):
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=False,
        args=[
            "--no-default-browser-check",
            "--no_sandbox",
            "--disable-blink-features=AutomationControlled",
        ],
        ignore_default_args=ignore_args,
        user_agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 720},
        bypass_csp=True,
        slow_mo=1000,
        chromium_sandbox=True,
        channel="chrome-dev",
    )
    return context


async def persistent_launch_async(
    playwright: Playwright, user_data_dir: str = "", record_video_dir="video"
):
    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
        ],
        ignore_default_args=ignore_args,
        user_agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        # viewport={"width": 1280, "height": 720},
        record_video_dir=record_video_dir,
        channel="chrome-dev",
        # slow_mo=1000,
    )
    return context


def next_free_port(port=9876, max_port=9999):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError("no free ports")


def connect_via_cdp(playwright: Playwright, user_data_dir: str = ""):
    cdp_address = (
        "ws://127.0.0.1:44677/devtools/browser/f41cafc7-7dfb-4b74-8e2a-d87a18751b07"
    )
    browser = playwright.chromium.connect_over_cdp(
        endpoint_url=cdp_address,
    )
    return browser


async def connect_via_cdp_async(playwright: Playwright, user_data_dir: str = ""):
    chrome_process = subprocess.Popen(
        [
            "/pw-browsers/chromium-1041/chrome-linux/chrome",
            # "--disable-dev-shm-usage",
            # "--no-startup-window",
            "--remote-debugging-port=0",
            f"--user-data-dir={user_data_dir}",
            "--disable-blink-features=AutomationControlled",
        ],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )
    for line in iter(chrome_process.stdout.readline, b""):
        if b"DevTools listening on" in line:
            cdp_address = line.split()[-1].decode("utf-8")
            break
    # cdp_address = (
    #     "ws://127.0.0.1:9876/devtools/browser/a175e70e-2b36-4b80-ba44-2be98b1d8f3f"
    # )
    browser = await playwright.chromium.connect_over_cdp(endpoint_url=cdp_address)
    return browser, chrome_process


def remove_extra_eol(text):
    # Replace EOL symbols
    text = text.replace("\n", " ")
    return re.sub(r"\s{2,}", " ", text)


def get_first_line(s):
    first_line = s.split("\n")[0]
    tokens = first_line.split()
    if len(tokens) > 8:
        return " ".join(tokens[:8]) + "..."
    else:
        return first_line


async def get_element_description(element, tag_name, role_value, type_value):
    """
    Asynchronously generates a descriptive text for a web element based on its tag type.
    Handles various HTML elements like 'select', 'input', and 'textarea', extracting attributes and content relevant to accessibility and interaction.
    """

    salient_attributes = [
        "alt",
        "aria-describedby",
        "aria-label",
        "aria-role",
        "input-checked",
        # "input-value",
        "label",
        "name",
        "option_selected",
        "placeholder",
        "readonly",
        "text-value",
        "title",
        "value",
    ]

    parent_value = "parent_node: "
    parent_locator = element.locator("xpath=..")
    num_parents = await parent_locator.count()
    if num_parents > 0:
        # only will be zero or one parent node
        parent_text = (await parent_locator.inner_text(timeout=0) or "").strip()
        if parent_text:
            parent_value += parent_text
    parent_value = remove_extra_eol(get_first_line(parent_value)).strip()
    if parent_value == "parent_node:":
        parent_value = ""
    else:
        parent_value += " "

    if tag_name == "select":
        text1 = "Selected Options: "
        text2 = ""
        text3 = " - Options: "
        text4 = ""

        text2 = await element.evaluate(
            "select => select.options[select.selectedIndex].textContent", timeout=0
        )

        if text2:
            options = await element.evaluate(
                "select => Array.from(select.options).map(option => option.text)",
                timeout=0,
            )
            text4 = " | ".join(options)

            if not text4:
                text4 = await element.text_content(timeout=0)
                if not text4:
                    text4 = await element.inner_text(timeout=0)

            return (
                parent_value + text1 + remove_extra_eol(text2.strip()) + text3 + text4
            )

    input_value = ""

    none_input_type = ["submit", "reset", "checkbox", "radio", "button", "file"]

    if tag_name == "input" or tag_name == "textarea":
        if role_value not in none_input_type and type_value not in none_input_type:
            text1 = "input value="
            text2 = await element.input_value(timeout=0)
            if text2:
                input_value = text1 + '"' + text2 + '"' + " "
    try:
        text_content = await element.text_content(timeout=5000)
    except PlaywrightError as e:
        print(f"Error occurred while getting text content: {e}")
    text = (text_content or "").strip()
    if text:
        text = remove_extra_eol(text)
        if len(text) > 80:
            text_content_in = await element.inner_text(timeout=0)
            text_in = (text_content_in or "").strip()
            if text_in:
                return input_value + remove_extra_eol(text_in)
        else:
            return input_value + text

    # get salient_attributes
    text1 = ""
    for attr in salient_attributes:
        attribute_value = await element.get_attribute(attr, timeout=0)
        if attribute_value:
            text1 += f"{attr}=" + '"' + attribute_value.strip() + '"' + " "

    text = (parent_value + text1).strip()
    if text:
        return input_value + remove_extra_eol(text.strip())

    # try to get from the first child node
    first_child_locator = element.locator("xpath=./child::*[1]")

    num_childs = await first_child_locator.count()
    if num_childs > 0:
        for attr in salient_attributes:
            attribute_value = await first_child_locator.get_attribute(attr, timeout=0)
            if attribute_value:
                text1 += f"{attr}=" + '"' + attribute_value.strip() + '"' + " "

        text = (parent_value + text1).strip()
        if text:
            return input_value + remove_extra_eol(text.strip())

    return None


async def get_element_data(element, tag_name):
    tag_name_list = ["a", "button", "input", "select", "textarea", "adc-tab"]

    try:
        # 1. Check element connection and visibility
        if not await element.evaluate("el => !!el.isConnected", timeout=500):
            return None
        is_hidden = await element.is_hidden(timeout=500)
        is_disabled = await element.is_disabled(timeout=500)
        if is_hidden or is_disabled:
            return None
    except Exception as e:
        print(f"[Element Check Error] {e}")
        return None

    # 2. Tag check
    tag_head = ""
    real_tag_name = ""
    if tag_name in tag_name_list:
        tag_head = tag_name
        real_tag_name = tag_name
    else:
        try:
            real_tag_name = await element.evaluate(
                "el => el.tagName.toLowerCase()", timeout=1000
            )
        except Exception as e:
            print(f"[tagName eval error] {e}")
            return None

        if real_tag_name in tag_name_list:
            return None
        tag_head = real_tag_name

    # 3. Attribute fetching (setting timeout=0 here may cause problems, it is recommended to set a small timeout)
    try:
        role_value = await element.get_attribute("role", timeout=500)
        type_value = await element.get_attribute("type", timeout=500)
    except Exception as e:
        print(f"[Attribute fetch error] {e}")
        return None

    # 4. Get description, with timeout protection
    try:
        description = await asyncio.wait_for(
            get_element_description(element, real_tag_name, role_value, type_value),
            timeout=3,  # Wait at most 3 seconds
        )
    except asyncio.TimeoutError:
        print("[Timeout] get_element_description timed out")
        return None
    except Exception as e:
        print(f"[Error] get_element_description error: {e}")
        return None

    if not description:
        return None

    # 5. Get bounding box, with timeout protection
    try:
        rect = await asyncio.wait_for(element.bounding_box(), timeout=2)
        if not rect:
            rect = {"x": 0, "y": 0, "width": 0, "height": 0}
    except Exception as e:
        print(f"[bounding_box error] {e}")
        rect = {"x": 0, "y": 0, "width": 0, "height": 0}

    # 6. Construct return value
    if role_value:
        tag_head += f' role="{role_value}"'
    if type_value:
        tag_head += f' type="{type_value}"'

    box_model = [
        rect["x"],
        rect["y"],
        rect["x"] + rect["width"],
        rect["y"] + rect["height"],
    ]
    center_point = (
        (box_model[0] + box_model[2]) / 2,
        (box_model[1] + box_model[3]) / 2,
    )

    return [center_point, description, tag_head, box_model, element, real_tag_name]


async def run_with_limit(element, tag_name, semaphore):
    async with semaphore:
        try:
            return await asyncio.wait_for(
                get_element_data(element, tag_name), timeout=5
            )
        except asyncio.TimeoutError:
            print("[Timeout] get_element_data took too long.")
            return None
        except Exception as e:
            print(f"[Error] get_element_data error: {e}")
            return None


async def get_interactive_elements_with_playwright(page):
    interactive_elements_selectors = [
        "a",
        "button",
        "div.footWrap--LePfCZWd > div > div.LeftButtonListForEmphasize--eCGPFYh5 > div:nth-child(2)",
        "#submitOrder",
        "input",
        'button:text("Add to cart")',
        "select",
        "textarea",
        "adc-tab",
        '[role="button"]',
        '[role="radio"]',
        '[role="option"]',
        '[role="combobox"]',
        '[role="textbox"]',
        '[role="listbox"]',
        '[role="menu"]',
        '[type="button"]',
        '[type="radio"]',
        '[type="combobox"]',
        '[type="textbox"]',
        '[type="listbox"]',
        '[type="menu"]',
        '[tabindex]:not([tabindex="-1"])',
        '[contenteditable]:not([contenteditable="false"])',
        "[onclick]",
        "[onfocus]",
        "[onkeydown]",
        "[onkeypress]",
        "[onkeyup]",
        "[checkbox]",
        '[aria-disabled="false"],[data-link]',
    ]

    semaphore = asyncio.Semaphore(50)
    tasks = []

    seen_elements = set()
    # await page.wait_for_load_state("domcontentloaded")
    for selector in interactive_elements_selectors:
        locator = page.locator(selector)
        try:
            element_count = await locator.count()
        except Exception as e:
            print(f"Failed to get element count for selector '{selector}': {e}")
            element_count = 0
        for index in range(element_count):
            element = locator.nth(index)
            tag_name = selector.replace(':not([tabindex="-1"])', "")
            tag_name = tag_name.replace(':not([contenteditable="false"])', "")
            task = run_with_limit(element, tag_name, semaphore)
            tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=False)

    interactive_elements = []
    for i in results:
        if i:
            if i[0] in seen_elements:
                continue
            else:
                seen_elements.add(i[0])
                interactive_elements.append(i)
    return interactive_elements


async def select_option(selector, value):
    best_option = [-1, "", -1]
    for i in range(await selector.locator("option").count()):
        option = await selector.locator("option").nth(i).inner_text()
        similarity = SequenceMatcher(None, option, value).ratio()
        if similarity > best_option[2]:
            best_option = [i, option, similarity]
    await selector.select_option(index=best_option[0], timeout=10000)
    return remove_extra_eol(best_option[1]).strip()


def saveconfig(config, save_file):
    """
    config is a dictionary.
    save_path: saving path include file name.
    """
    if isinstance(save_file, str):
        save_file = Path(save_file)
    if isinstance(config, dict):
        with open(save_file, "w") as f:
            config_without_key = config
            if "openai" in config_without_key.keys():
                config_without_key["openai"]["api_key"] = "Your API key here"
            toml.dump(config_without_key, f)
    else:
        os.system(" ".join(["cp", str(config), str(save_file)]))
