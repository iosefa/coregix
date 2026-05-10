# Advanced Registration

These utilities expose lower-level transform estimation and application functions. They are useful for custom registration experiments, but most users should use `coregix.align_image_pair()` instead.

::: coregix.preprocess.registration
    options:
      members:
        - estimate_elastix_transform
        - apply_elastix_transform
        - apply_elastix_transform_array
        - deformation_field_from_transform
        - deformation_field_from_transform_region
        - run_elastix_registration
