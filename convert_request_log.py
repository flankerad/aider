import json
import re
import sys
import os

def convert_log_to_jsonl(input_filename="verbose_python_requests.md", output_filename="verbose_python_requests.jsonl"):
    """
    Converts the old mixed text/JSON log format to JSON Lines format.
    """
    if not os.path.exists(input_filename):
        print(f"Input file '{input_filename}' not found. Skipping conversion.", file=sys.stderr)
        return

    print(f"Converting '{input_filename}' to JSON Lines format in '{output_filename}'...")
    entries_converted = 0
    errors = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:

            content = infile.read()
            # Split based on the header line, keeping the delimiter part
            # Use a regex that captures the timestamp from the header
            log_chunks = re.split(r'(---\s*LLM Request\s*(\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2})\s*---)\n', content)

            # The first element might be empty or contain text before the first log
            # Process elements in steps of 3: [separator, timestamp, json_block]
            i = 1
            while i < len(log_chunks):
                separator = log_chunks[i]
                timestamp_from_header = log_chunks[i+1]
                json_block_str = log_chunks[i+2].strip()
                i += 3

                if not json_block_str:
                    continue

                try:
                    # Parse the pretty-printed JSON block
                    parsed_entry = json.loads(json_block_str)

                    # Ensure the structure matches what we expect (optional but good practice)
                    if isinstance(parsed_entry, dict) and "timestamp" in parsed_entry and "model" in parsed_entry and "request_kwargs" in parsed_entry:
                        # Use the timestamp from the parsed JSON if available, otherwise fallback?
                        # Let's trust the parsed JSON's timestamp field primarily.
                        # Or we could enforce using timestamp_from_header if needed.

                        # Serialize back to a compact JSON string (JSON Line)
                        json_line = json.dumps(parsed_entry, ensure_ascii=False)
                        outfile.write(json_line + '\n')
                        entries_converted += 1
                    else:
                        print(f"Warning: Skipping entry with unexpected structure near timestamp {timestamp_from_header}", file=sys.stderr)
                        errors += 1

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON entry near timestamp {timestamp_from_header}. Error: {e}", file=sys.stderr)
                    # Optionally log the problematic block:
                    # print(f"Problematic block:\n{json_block_str[:200]}...", file=sys.stderr)
                    errors += 1
                except Exception as e:
                     print(f"Warning: Unexpected error processing entry near timestamp {timestamp_from_header}. Error: {e}", file=sys.stderr)
                     errors += 1


    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.", file=sys.stderr)
        return
    except IOError as e:
        print(f"Error reading or writing file: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return

    print(f"Conversion complete. Converted {entries_converted} entries.")
    if errors > 0:
        print(f"Encountered {errors} errors/warnings during conversion.")

if __name__ == "__main__":
    # You can optionally add command-line argument parsing here
    # to specify input/output files if needed.
    convert_log_to_jsonl()
    print("\nInstructions:")
    print(f"1. Review the new file 'verbose_python_requests.jsonl'.")
    print(f"2. If the conversion looks correct, you can optionally replace")
    print(f"   'verbose_python_requests.md' with 'verbose_python_requests.jsonl'.")
    print(f"   Make sure to rename 'verbose_python_requests.jsonl' to")
    print(f"   'verbose_python_requests.md' if you want aider to continue logging to it.")
    print(f"   Example (use with caution):")
    print(f"   mv verbose_python_requests.jsonl verbose_python_requests.md")
