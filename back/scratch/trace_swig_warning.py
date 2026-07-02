import warnings

# Trace imports
class ImportTracer:
    def __init__(self):
        self.depth = 0

    def __call__(self, name, globals=None, locals=None, fromlist=None, level=0):
        # Call the original import
        self.depth += 1
        try:
            module = original_import(name, globals, locals, fromlist, level)
            return module
        finally:
            self.depth -= 1

# Enable deprecation warnings and print their traceback
def warning_handler(message, category, filename, lineno, file=None, line=None):
    if "SwigPyPacked" in str(message) or "SwigPyObject" in str(message) or "swigvarlink" in str(message):
        print(f"\n[Warning] {category.__name__}: {message}")
        print(f"  at {filename}:{lineno}")
        # Print active modules in import stack
        import traceback
        traceback.print_stack()

warnings.showwarning = warning_handler
warnings.simplefilter("always", DeprecationWarning)

# Let's import key libraries to see which one triggers it
try:
    print("Importing usearch...")
except Exception as e:
    print("usearch import failed:", e)

try:
    print("Importing ebooklib...")
except Exception as e:
    print("ebooklib import failed:", e)

try:
    print("Importing curl_cffi...")
except Exception as e:
    print("curl_cffi import failed:", e)

try:
    print("Importing spacy...")
except Exception as e:
    print("spacy import failed:", e)

try:
    print("Importing faster_whisper...")
except Exception as e:
    print("faster_whisper import failed:", e)

try:
    print("Importing marker...")
except Exception as e:
    print("marker import failed:", e)
