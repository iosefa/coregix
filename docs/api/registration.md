# Advanced Registration

These utilities expose lower-level transform estimation and application functions. They are useful for custom registration experiments, but most users should use `coregix.align_image_pair()` instead. Coregix transform JSON sidecars should be applied with `coregix.apply_coregix_transform()` or `apply-coregix-transform`, not the low-level `apply_elastix_transform()` helper.

::: coregix.preprocess.registration
    options:
      members:
        - estimate_elastix_transform
        - apply_elastix_transform
        - apply_elastix_transform_array
        - deformation_field_from_transform
        - deformation_field_from_transform_region
        - run_elastix_registration
