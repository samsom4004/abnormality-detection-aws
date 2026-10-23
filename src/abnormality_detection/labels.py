"""The 14 labelled findings (Kvasir-Capsule style) and the anomaly view built on top of them.

``NORMAL_LIKE`` groups anatomical landmarks and clean mucosa. The *abnormal probability* of an image is
the probability mass on every class that is **not** normal-like. This turns the 14-class classifier into an
anomaly detector without training a second model. Clinical teams should review this grouping.
"""

CLASSES = [
    "Angiectasia",
    "Blood - fresh",
    "Blood - hematin",
    "Erosion",
    "Erythema",
    "Foreign body",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Polyp",
    "Pylorus",
    "Reduced mucosal view",
    "ampulla_of_vater",
    "ulcer",
]

NORMAL_LIKE = [
    "Normal clean mucosa",
    "Ileocecal valve",
    "Pylorus",
    "ampulla_of_vater",
    "Reduced mucosal view",
]

# images where the view itself is poor: always route to a human
LOW_QUALITY = ["Reduced mucosal view"]

DISCLAIMER = (
    "Research prototype. Not a medical device and not for clinical use. "
    "A qualified clinician must review every image."
)


def abnormal_indices(classes=CLASSES, normal_like=NORMAL_LIKE):
    return [i for i, c in enumerate(classes) if c not in set(normal_like)]
