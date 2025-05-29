"""
Multilooking implementation for SAR complex images.

This module provides functions for multilooking SAR complex images to reduce
speckle noise by spatial averaging. Multilooking is a standard technique in
SAR processing where neighboring pixels are averaged to improve the signal-to-noise
ratio at the cost of spatial resolution.

Author: GitHub Copilot
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


def multilook_complex(
    complex_image: np.ndarray,
    num_looks: Union[int, Tuple[int, int]],
    method: str = "average",
    preserve_phase: bool = True
) -> np.ndarray:
    """
    Apply multilooking to a complex SAR image.
    
    Multilooking reduces speckle noise by averaging neighboring pixels in the
    complex image. This function supports different averaging methods and can
    preserve phase information.
    
    Args:
        complex_image (np.ndarray): Input complex SAR image (2D complex array).
        num_looks (Union[int, Tuple[int, int]]): Number of looks to apply.
            If int, same number of looks in both dimensions.
            If tuple, (azimuth_looks, range_looks).
        method (str, optional): Multilooking method. Options:
            - "average": Simple complex averaging (default)
            - "intensity": Average intensities then take square root
            - "coherent": Coherent averaging preserving phase
        preserve_phase (bool, optional): Whether to preserve relative phase.
            Only applicable for "average" method. Defaults to True.
    
    Returns:
        np.ndarray: Multilooked complex image with reduced dimensions.
        
    Raises:
        ValueError: If input parameters are invalid.
        
    Examples:
        >>> # Simple 3x3 multilooking
        >>> multilooked = multilook_complex(complex_sar, 3)
        
        >>> # Different looks in azimuth and range
        >>> multilooked = multilook_complex(complex_sar, (2, 4))
        
        >>> # Intensity-based multilooking
        >>> multilooked = multilook_complex(complex_sar, 3, method="intensity")
    """
    if complex_image.ndim != 2:
        raise ValueError("Input complex_image must be 2D")
    
    if not np.iscomplexobj(complex_image):
        warnings.warn("Input image is not complex. Converting to complex.")
        complex_image = complex_image.astype(complex)
    
    # Handle num_looks parameter
    if isinstance(num_looks, int):
        if num_looks < 1:
            raise ValueError("num_looks must be positive")
        azimuth_looks = range_looks = num_looks
    elif isinstance(num_looks, (tuple, list)) and len(num_looks) == 2:
        azimuth_looks, range_looks = num_looks
        if azimuth_looks < 1 or range_looks < 1:
            raise ValueError("Both look values must be positive")
    else:
        raise ValueError("num_looks must be int or tuple of 2 ints")
    
    # Check if image dimensions are compatible
    height, width = complex_image.shape
    if height < azimuth_looks or width < range_looks:
        raise ValueError(f"Image dimensions {complex_image.shape} too small for "
                        f"multilooking with {azimuth_looks}x{range_looks} looks")
    
    # Calculate output dimensions
    new_height = height // azimuth_looks
    new_width = width // range_looks
    
    # Crop to ensure even division
    cropped_height = new_height * azimuth_looks
    cropped_width = new_width * range_looks
    cropped_image = complex_image[:cropped_height, :cropped_width]
    
    if method == "average":
        return _multilook_average(cropped_image, azimuth_looks, range_looks, preserve_phase)
    elif method == "intensity":
        return _multilook_intensity(cropped_image, azimuth_looks, range_looks)
    elif method == "coherent":
        return _multilook_coherent(cropped_image, azimuth_looks, range_looks)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'average', 'intensity', or 'coherent'")


def _multilook_average(
    complex_image: np.ndarray,
    azimuth_looks: int,
    range_looks: int,
    preserve_phase: bool
) -> np.ndarray:
    """Average-based multilooking implementation."""
    height, width = complex_image.shape
    new_height = height // azimuth_looks
    new_width = width // range_looks
    
    # Reshape for block averaging
    reshaped = complex_image.reshape(
        new_height, azimuth_looks, new_width, range_looks
    )
    
    if preserve_phase:
        # Compute magnitude and phase separately
        magnitude = np.abs(reshaped)
        phase = np.angle(reshaped)
        
        # Average magnitude
        avg_magnitude = np.mean(magnitude, axis=(1, 3))
        
        # Average phase (handling phase wrapping)
        # Convert to complex unit vectors and average
        unit_vectors = np.exp(1j * phase)
        avg_unit_vector = np.mean(unit_vectors, axis=(1, 3))
        avg_phase = np.angle(avg_unit_vector)
        
        # Reconstruct complex image
        result = avg_magnitude * np.exp(1j * avg_phase)
    else:
        # Simple complex averaging
        result = np.mean(reshaped, axis=(1, 3))
    
    return result


def _multilook_intensity(
    complex_image: np.ndarray,
    azimuth_looks: int,
    range_looks: int
) -> np.ndarray:
    """Intensity-based multilooking implementation."""
    height, width = complex_image.shape
    new_height = height // azimuth_looks
    new_width = width // range_looks
    
    # Compute intensity
    intensity = np.abs(complex_image) ** 2
    
    # Reshape for block averaging
    reshaped = intensity.reshape(
        new_height, azimuth_looks, new_width, range_looks
    )
    
    # Average intensity
    avg_intensity = np.mean(reshaped, axis=(1, 3))
    
    # Convert back to complex (magnitude only, phase=0)
    return np.sqrt(avg_intensity).astype(complex)


def _multilook_coherent(
    complex_image: np.ndarray,
    azimuth_looks: int,
    range_looks: int
) -> np.ndarray:
    """Coherent multilooking implementation."""
    height, width = complex_image.shape
    new_height = height // azimuth_looks
    new_width = width // range_looks
    
    # Reshape for block averaging
    reshaped = complex_image.reshape(
        new_height, azimuth_looks, new_width, range_looks
    )
    
    # Coherent averaging (simple complex mean)
    return np.mean(reshaped, axis=(1, 3))


def multilook_separate_channels(
    real_part: np.ndarray,
    imaginary_part: np.ndarray,
    num_looks: Union[int, Tuple[int, int]],
    method: str = "average",
    preserve_phase: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multilooking to separate real and imaginary parts of SAR image.
    
    This function is useful when working with real and imaginary parts stored
    as separate arrays, as commonly found in SAR processing pipelines.
    
    Args:
        real_part (np.ndarray): Real component of complex SAR image (2D).
        imaginary_part (np.ndarray): Imaginary component of complex SAR image (2D).
        num_looks (Union[int, Tuple[int, int]]): Number of looks to apply.
        method (str, optional): Multilooking method. Defaults to "average".
        preserve_phase (bool, optional): Whether to preserve phase. Defaults to True.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Multilooked (real_part, imaginary_part).
        
    Raises:
        ValueError: If input arrays have different shapes.
        
    Examples:
        >>> real_ml, imag_ml = multilook_separate_channels(Re, Im, 3)
        >>> real_ml, imag_ml = multilook_separate_channels(Re, Im, (2, 4))
    """
    if real_part.shape != imaginary_part.shape:
        raise ValueError("Real and imaginary parts must have the same shape")
    
    # Combine into complex image
    complex_image = real_part + 1j * imaginary_part
    
    # Apply multilooking
    multilooked_complex = multilook_complex(
        complex_image, num_looks, method, preserve_phase
    )
    
    # Return separate components
    return np.real(multilooked_complex), np.imag(multilooked_complex)


def adaptive_multilook(
    complex_image: np.ndarray,
    target_looks: int,
    speckle_threshold: float = 0.5,
    min_looks: int = 1,
    max_looks: Optional[int] = None
) -> np.ndarray:
    """
    Apply adaptive multilooking based on local speckle characteristics.
    
    This function adaptively adjusts the number of looks based on local
    speckle statistics, applying more averaging in areas with high speckle
    and preserving resolution in areas with low speckle.
    
    Args:
        complex_image (np.ndarray): Input complex SAR image (2D).
        target_looks (int): Target number of looks for homogeneous areas.
        speckle_threshold (float, optional): Speckle variance threshold for
            adaptive adjustment. Defaults to 0.5.
        min_looks (int, optional): Minimum number of looks. Defaults to 1.
        max_looks (Optional[int], optional): Maximum number of looks.
            If None, uses 2 * target_looks.
    
    Returns:
        np.ndarray: Adaptively multilooked complex image.
        
    Note:
        This is a simplified adaptive implementation. More sophisticated
        methods may use edge detection, texture analysis, or other metrics.
    """
    if max_looks is None:
        max_looks = 2 * target_looks
    
    # Compute local speckle statistics using a sliding window
    intensity = np.abs(complex_image) ** 2
    
    # Use a simple 5x5 window for speckle estimation
    from scipy.ndimage import uniform_filter
    
    local_mean = uniform_filter(intensity, size=5)
    local_var = uniform_filter(intensity**2, size=5) - local_mean**2
    
    # Avoid division by zero
    local_var = np.maximum(local_var, 1e-10)
    local_mean = np.maximum(local_mean, 1e-10)
    
    # Coefficient of variation as speckle measure
    speckle_index = np.sqrt(local_var) / local_mean
    
    # Determine adaptive number of looks
    # High speckle -> more looks, low speckle -> fewer looks
    adaptive_looks = np.where(
        speckle_index > speckle_threshold,
        max_looks,
        np.where(
            speckle_index > speckle_threshold / 2,
            target_looks,
            min_looks
        )
    )
    
    # For simplicity, use the most common look value
    # In practice, you might want to implement region-based processing
    most_common_looks = int(np.median(adaptive_looks))
    most_common_looks = np.clip(most_common_looks, min_looks, max_looks)
    
    return multilook_complex(complex_image, most_common_looks)


def multilook_with_overlap(
    complex_image: np.ndarray,
    num_looks: Union[int, Tuple[int, int]],
    overlap_factor: float = 0.5,
    method: str = "average"
) -> np.ndarray:
    """
    Apply overlapping multilooking to preserve more spatial information.
    
    Instead of non-overlapping blocks, this function uses overlapping windows
    to reduce the loss of spatial resolution while still achieving noise reduction.
    
    Args:
        complex_image (np.ndarray): Input complex SAR image (2D).
        num_looks (Union[int, Tuple[int, int]]): Number of looks to apply.
        overlap_factor (float, optional): Overlap factor between windows (0-1).
            0 = no overlap, 0.5 = 50% overlap. Defaults to 0.5.
        method (str, optional): Multilooking method. Defaults to "average".
    
    Returns:
        np.ndarray: Multilooked complex image with overlapping windows.
        
    Note:
        Output size will be larger than standard multilooking due to overlap.
    """
    if not 0 <= overlap_factor < 1:
        raise ValueError("overlap_factor must be in range [0, 1)")
    
    # Handle num_looks parameter
    if isinstance(num_looks, int):
        azimuth_looks = range_looks = num_looks
    else:
        azimuth_looks, range_looks = num_looks
    
    height, width = complex_image.shape
    
    # Calculate step sizes with overlap
    az_step = max(1, int(azimuth_looks * (1 - overlap_factor)))
    rg_step = max(1, int(range_looks * (1 - overlap_factor)))
    
    # Calculate output dimensions
    out_height = (height - azimuth_looks) // az_step + 1
    out_width = (width - range_looks) // rg_step + 1
    
    result = np.zeros((out_height, out_width), dtype=complex)
    
    for i in range(out_height):
        for j in range(out_width):
            # Extract window
            az_start = i * az_step
            az_end = az_start + azimuth_looks
            rg_start = j * rg_step
            rg_end = rg_start + range_looks
            
            window = complex_image[az_start:az_end, rg_start:rg_end]
            
            # Apply multilooking to window
            if method == "average":
                result[i, j] = np.mean(window)
            elif method == "intensity":
                intensity = np.abs(window) ** 2
                result[i, j] = np.sqrt(np.mean(intensity))
            elif method == "coherent":
                result[i, j] = np.mean(window)
    
    return result


def compute_equivalent_number_of_looks(multilooked_image: np.ndarray) -> float:
    """
    Estimate the equivalent number of looks (ENL) from a multilooked image.
    
    The ENL is an important quality metric for SAR images that indicates
    the effective amount of averaging that has been applied.
    
    Args:
        multilooked_image (np.ndarray): Multilooked complex or intensity image.
    
    Returns:
        float: Estimated equivalent number of looks.
        
    Note:
        ENL = mean²/variance for intensity images.
        For complex images, intensity is computed first.
    """
    if np.iscomplexobj(multilooked_image):
        intensity = np.abs(multilooked_image) ** 2
    else:
        intensity = multilooked_image
    
    # Remove zero values to avoid numerical issues
    intensity_nonzero = intensity[intensity > 0]
    
    if len(intensity_nonzero) == 0:
        return 0.0
    
    mean_intensity = np.mean(intensity_nonzero)
    var_intensity = np.var(intensity_nonzero)
    
    if var_intensity == 0:
        return float('inf')
    
    enl = mean_intensity ** 2 / var_intensity
    return enl


# Example usage and demonstration functions
def demo_multilooking():
    """
    Demonstrate different multilooking techniques on synthetic SAR data.
    """
    print("Multilooking Demo")
    print("================")
    
    # Create synthetic complex SAR data with speckle
    np.random.seed(42)
    height, width = 256, 256
    
    # Simulate a simple target in noisy background
    target = np.zeros((height, width), dtype=complex)
    target[120:140, 120:140] = 2.0 + 0.5j  # Bright target
    
    # Add speckle noise
    speckle_real = np.random.rayleigh(1.0, (height, width))
    speckle_imag = np.random.rayleigh(1.0, (height, width))
    speckle_phase = np.random.uniform(-np.pi, np.pi, (height, width))
    speckle = speckle_real * np.exp(1j * speckle_phase)
    
    # Combine target and speckle
    synthetic_sar = target + 0.5 * speckle
    
    print(f"Original image shape: {synthetic_sar.shape}")
    print(f"Original ENL: {compute_equivalent_number_of_looks(synthetic_sar):.2f}")
    
    # Apply different multilooking methods
    ml_3x3 = multilook_complex(synthetic_sar, 3)
    print(f"3x3 multilook shape: {ml_3x3.shape}")
    print(f"3x3 multilook ENL: {compute_equivalent_number_of_looks(ml_3x3):.2f}")
    
    ml_2x4 = multilook_complex(synthetic_sar, (2, 4))
    print(f"2x4 multilook shape: {ml_2x4.shape}")
    print(f"2x4 multilook ENL: {compute_equivalent_number_of_looks(ml_2x4):.2f}")
    
    ml_intensity = multilook_complex(synthetic_sar, 3, method="intensity")
    print(f"Intensity multilook ENL: {compute_equivalent_number_of_looks(ml_intensity):.2f}")
    
    adaptive_ml = adaptive_multilook(synthetic_sar, target_looks=3)
    print(f"Adaptive multilook shape: {adaptive_ml.shape}")
    print(f"Adaptive multilook ENL: {compute_equivalent_number_of_looks(adaptive_ml):.2f}")
    
    return synthetic_sar, ml_3x3, ml_2x4, ml_intensity, adaptive_ml


if __name__ == "__main__":
    # Run demonstration
    demo_results = demo_multilooking()
    print("\nMultilooking implementation complete!")
    print("Functions available:")
    print("- multilook_complex(): Main multilooking function")
    print("- multilook_separate_channels(): For separate Re/Im arrays")
    print("- adaptive_multilook(): Adaptive multilooking")
    print("- multilook_with_overlap(): Overlapping window multilooking")
    print("- compute_equivalent_number_of_looks(): ENL computation")