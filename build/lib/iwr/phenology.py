from dataclasses import dataclass, field
from numpy.typing import NDArray
import numpy as np
import xarray as xr
from datetime import datetime
from iwr.crops import Crop

PHENOLOGY_INACTIVE = 0
PHENOLOGY_GROWING = 1
PHENOLOGY_MAXIMUM = 2
PHENOLOGY_SENESCENCE = 3

@dataclass
class Phenology:
    """
    Phenology representation based on ASAP/SPAM phenology rasters.

    Phenological states:
        0 = inactive
        1 = growing
        2 = maximum
        3 = senescence

    ASAP phenology dates are represented on an extended 1-108 dekad
    calendar. For each calendar dekad (1-36), the corresponding
    positions d, d+36 and d+72 are therefore tested.
    """

    valid_mask: NDArray[np.uint8] | None = None
    time_index: NDArray[np.datetime64] | None = None

    crop: Crop | None = None
    sowing_date: datetime | None = None

    # Phenology rasters
    phenoe1: NDArray[np.uint8] | None = None
    phenoe2: NDArray[np.uint8] | None = None

    phenom1: NDArray[np.uint8] | None = None
    phenom2: NDArray[np.uint8] | None = None

    phenonseasons: NDArray[np.uint8] | None = None

    phenos1: NDArray[np.uint8] | None = None
    phenos2: NDArray[np.uint8] | None = None

    phenosen1: NDArray[np.uint8] | None = None
    phenosen2: NDArray[np.uint8] | None = None

    state: xr.DataArray = field(init=False)
    Kc_matrix: xr.DataArray = field(init=False)

    inactive_kc: float = 0.0
    

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self):

        if self.valid_mask is None:
            raise ValueError(
                "valid_mask is required to initialize phenology state"
            )

        if self.valid_mask.ndim != 2:
            raise ValueError(
                f"valid_mask must be a 2-D array, got shape "
                f"{self.valid_mask.shape}"
            )

        rows, cols = self.valid_mask.shape

        # Check that every supplied phenology raster has the same shape
        # as the valid mask.
        raster_names = (
            "phenoe1",
            "phenoe2",
            "phenom1",
            "phenom2",
            "phenonseasons",
            "phenos1",
            "phenos2",
            "phenosen1",
            "phenosen2",
        )

        for name in raster_names:

            raster = getattr(self, name)

            if raster is not None and raster.shape != self.valid_mask.shape:
                raise ValueError(
                    f"{name} shape {raster.shape} does not match "
                    f"valid_mask shape {self.valid_mask.shape}"
                )

        # ASAP calendar dekads are numbered 1-36.
        self.state = xr.DataArray(
            np.zeros(
                (36, rows, cols),
                dtype=np.uint8,
            ),
            dims=("time", "row", "col"),
            coords={
                "time": np.arange(1, 37),
                "row": np.arange(rows),
                "col": np.arange(cols),
            },
            name="phenology_state",
            attrs={
                "legend": {
                    PHENOLOGY_INACTIVE: "PHENOLOGY_INACTIVE",
                    PHENOLOGY_GROWING: "PHENOLOGY_GROWING",
                    PHENOLOGY_MAXIMUM: "PHENOLOGY_MAXIMUM",
                    PHENOLOGY_SENESCENCE: "PHENOLOGY_SENESCENCE",
                }
            },
        )

        # Kc matrix will be built based on the crop's phenological parameters.
        # Initialize Ks array with ones (no stress)
        self.Kc_matrix = xr.DataArray(
            np.full(
                (366, rows, cols),
                self.inactive_kc,
                dtype=np.float32,
            ),
            dims=("time", "row", "col"),
            coords={
                "time": np.arange(1, 367),
                "row": np.arange(rows),
                "col": np.arange(cols),
            },
            name="Kc",
            attrs={
                "long_name": "daily crop coefficient",
            },
        )



    # ------------------------------------------------------------------
    # Construction from dates
    # ------------------------------------------------------------------

    @classmethod
    def from_dates(cls):
        raise NotImplementedError(
            "from_dates method is not implemented yet"
        )

    # ------------------------------------------------------------------
    # Internal utility
    # ------------------------------------------------------------------

    @staticmethod
    def _update_status_for_one_season(
        status,
        current_dekad,
        sos,
        tom,
        sen,
        eos,
        valid_mask,
    ):
        """
        Update phenological state for one growing season.

        ASAP phenology values use the extended 1-108 dekad calendar.

        For calendar dekad d, the three possible positions are:

            d
            d + 36
            d + 72

        Phenological phases are defined as:

            SOS <= d < TOM      -> growing
            TOM <= d < SEN      -> maximum
            SEN <= d <= EOS     -> senescence

        Parameters
        ----------
        status : ndarray
            Current phenology state.

        current_dekad : int
            Calendar dekad, from 1 to 36.

        sos : ndarray
            Start of season.

        tom : ndarray
            Time of maximum vegetation.

        sen : ndarray
            Start of senescence.

        eos : ndarray
            End of season.

        valid_mask : ndarray of bool
            Pixels where this season exists and should be evaluated.

        Returns
        -------
        ndarray
            Updated phenology status.
        """

        # Ensure that stage ordering itself is valid.
        stage_valid = (
            valid_mask
            & (sos <= tom)
            & (tom <= sen)
            & (sen <= eos)
        )

        current_dekad_candidates = (
            current_dekad,
            current_dekad + 36,
            current_dekad + 72,
        )

        for current_v in current_dekad_candidates:

            # ----------------------------------------------------------
            # Is this season active at the current extended-calendar
            # position?
            # ----------------------------------------------------------

            active_mask = (
                stage_valid
                & (current_v >= sos)
                & (current_v <= eos)
            )

            # ----------------------------------------------------------
            # Growing phase
            #
            # SOS included
            # TOM excluded
            # ----------------------------------------------------------

            growing_mask = (
                active_mask
                & (current_v >= sos)
                & (current_v < tom)
            )

            # ----------------------------------------------------------
            # Maximum phase
            #
            # TOM included
            # SEN excluded
            # ----------------------------------------------------------

            maximum_mask = (
                active_mask
                & (current_v >= tom)
                & (current_v < sen)
            )

            # ----------------------------------------------------------
            # Senescence phase
            #
            # SEN and EOS included
            # ----------------------------------------------------------

            senescence_mask = (
                active_mask
                & (current_v >= sen)
                & (current_v <= eos)
            )

            # IMPORTANT:
            #
            # Only active phases overwrite status.
            #
            # We DO NOT set inactive pixels to zero here, because another
            # candidate (d+36 or d+72) may already have identified this
            # pixel as active.
            #
            # Similarly, season 1 may already have set a valid state
            # before season 2 is evaluated.

            status[growing_mask] = 1
            status[maximum_mask] = 2
            status[senescence_mask] = 3

        return status

    # ------------------------------------------------------------------
    # Construction from SPAM / ASAP phenology layers
    # ------------------------------------------------------------------

    @classmethod
    def from_spam(
        cls,
        valid_mask,
        phenoe1,
        phenoe2,
        phenom1,
        phenom2,
        phenonseasons,
        phenos1,
        phenos2,
        phenosen1,
        phenosen2,
    ):
        """
        Build the complete 36-dekad phenology state cube from
        ASAP/SPAM phenology rasters.

        Parameters
        ----------
        valid_mask : ndarray
            Spatial validity mask. Non-zero values are interpreted
            as valid pixels.

        phenoe1, phenoe2 : ndarray
            End of season for seasons 1 and 2.

        phenom1, phenom2 : ndarray
            Time of maximum vegetation for seasons 1 and 2.

        phenonseasons : ndarray
            Number of growing seasons at each pixel.

        phenos1, phenos2 : ndarray
            Start of season for seasons 1 and 2.

        phenosen1, phenosen2 : ndarray
            Start of senescence for seasons 1 and 2.

        Returns
        -------
        xr.DataArray
            Phenological states with dimensions:

                time = 36
                row
                col

            Values:

                0 = inactive
                1 = growing
                2 = maximum
                3 = senescence
        """

        phenology = cls(
            valid_mask=valid_mask,
            phenoe1=phenoe1,
            phenoe2=phenoe2,
            phenom1=phenom1,
            phenom2=phenom2,
            phenonseasons=phenonseasons,
            phenos1=phenos1,
            phenos2=phenos2,
            phenosen1=phenosen1,
            phenosen2=phenosen2,
        )

        # --------------------------------------------------------------
        # Basic validity
        # --------------------------------------------------------------

        base_valid = phenology.valid_mask.astype(bool)

        number_of_seasons = phenology.phenonseasons

        # --------------------------------------------------------------
        # Season existence masks
        # --------------------------------------------------------------

        season_1_mask = (
            base_valid
            & (number_of_seasons >= 1)
        )

        season_2_mask = (
            base_valid
            & (number_of_seasons >= 2)
        )

        # --------------------------------------------------------------
        # Process the 36 calendar dekads
        # --------------------------------------------------------------

        for curr_dekad in phenology.state.time.values:

            #print(f"Processing dekad {curr_dekad}...")

            # Start every calendar dekad as inactive.
            #
            # Active phases found in either season will overwrite this.
            status = np.zeros(
                phenology.valid_mask.shape,
                dtype=np.uint8,
            )

            # ----------------------------------------------------------
            # Season 1
            # ----------------------------------------------------------

            status = phenology._update_status_for_one_season(
                status=status,
                current_dekad=curr_dekad,
                sos=phenology.phenos1,
                tom=phenology.phenom1,
                sen=phenology.phenosen1,
                eos=phenology.phenoe1,
                valid_mask=season_1_mask,
            )

            # ----------------------------------------------------------
            # Season 2
            # ----------------------------------------------------------

            status = phenology._update_status_for_one_season(
                status=status,
                current_dekad=curr_dekad,
                sos=phenology.phenos2,
                tom=phenology.phenom2,
                sen=phenology.phenosen2,
                eos=phenology.phenoe2,
                valid_mask=season_2_mask,
            )

            # ----------------------------------------------------------
            # Store result
            # ----------------------------------------------------------

            phenology.state.loc[
                dict(time=curr_dekad)
            ] = status

        return phenology
    
    
    def _state_to_daily(self) -> NDArray[np.uint8]:
        """
        Expand the 36-dekad phenology state cube to a daily 366-day calendar.

        Dekads are interpreted as:
            days 1-10
            days 11-20
            days 21-end of month

        Returns
        -------
        NDArray[np.uint8]
            Daily phenology state with shape:
            (366, rows, cols)
        """

        month_lengths = (
            31, 29, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31,
        )

        dekad_lengths = []

        for month_length in month_lengths:
            dekad_lengths.extend(
                (
                    10,
                    10,
                    max(month_length - 20, 0),
                )
            )

        dekad_lengths = np.asarray(
            dekad_lengths,
            dtype=np.int16,
        )

        daily_dekad_index = np.repeat(
            np.arange(36),
            dekad_lengths,
        )

        if daily_dekad_index.size != 366:
            raise RuntimeError(
                "Internal error: daily dekad index does not contain 366 days."
            )

        return self.state.values[daily_dekad_index]


    @staticmethod
    def _interpolate_stage(
        stage_mask: NDArray[np.bool_],
        start_value: float,
        end_value: float,
    ) -> NDArray[np.float32]:
        """
        Calculate linear interpolation within each contiguous occurrence
        of a phenological stage.

        The time dimension is treated as circular, allowing a phenological
        stage to continue across December -> January.

        Parameters
        ----------
        stage_mask : ndarray of bool
            Shape (365, rows, cols).
            True where the requested phenological stage is active.

        start_value : float
            Kc at the beginning of the stage.

        end_value : float
            Kc at the end of the stage.

        Returns
        -------
        ndarray of float32
            Interpolated values for the stage.
            Values outside the stage are zero.
        """

        n_days = stage_mask.shape[0]

        # --------------------------------------------------------------
        # Count how many consecutive days have elapsed since the
        # beginning of the current stage.
        #
        # We perform two passes over the year so that a stage beginning
        # in December and continuing into January is handled correctly.
        # --------------------------------------------------------------

        days_since_start = np.zeros(
            stage_mask.shape,
            dtype=np.uint16,
        )

        counter = np.zeros(
            stage_mask.shape[1:],
            dtype=np.uint16,
        )

        for i in range(2 * n_days):

            t = i % n_days
            active = stage_mask[t]

            counter = np.where(
                active,
                np.minimum(counter + 1, n_days),
                0,
            )

            if i >= n_days:
                days_since_start[i - n_days] = counter

        # --------------------------------------------------------------
        # Count how many days remain until the end of the stage.
        #
        # Again, two passes are used to correctly handle stages crossing
        # the end of the year.
        # --------------------------------------------------------------

        days_to_end = np.zeros(
            stage_mask.shape,
            dtype=np.uint16,
        )

        counter.fill(0)

        for i in range(2 * n_days - 1, -1, -1):

            t = i % n_days
            active = stage_mask[t]

            counter = np.where(
                active,
                np.minimum(counter + 1, n_days),
                0,
            )

            if i < n_days:
                days_to_end[i] = counter

        # Total number of days belonging to the contiguous stage.
        #
        # Cast to int32 before subtracting 1 to avoid unsigned integer
        # underflow outside the active stage.
        stage_length = (
            days_since_start.astype(np.int32)
            + days_to_end.astype(np.int32)
            - 1
        )

        # --------------------------------------------------------------
        # Relative position within the stage
        #
        # first day -> 0
        # last day  -> 1
        # --------------------------------------------------------------

        progress = np.zeros(
            stage_mask.shape,
            dtype=np.float32,
        )

        multi_day = (
            stage_mask
            & (stage_length > 1)
        )

        progress[multi_day] = (
            (
                days_since_start[multi_day].astype(np.float32)
                - 1.0
            )
            /
            (
                stage_length[multi_day].astype(np.float32)
                - 1.0
            )
        )

        # Defensive handling for a stage consisting of only one day.
        single_day = (
            stage_mask
            & (stage_length == 1)
        )

        progress[single_day] = 1.0

        # Very unusual case: a pixel remains in the same stage for the
        # complete year. There is no natural boundary, so interpolate
        # from day 1 to day 365.
        full_year = stage_mask.all(axis=0)

        if np.any(full_year):

            annual_progress = np.linspace(
                0.0,
                1.0,
                n_days,
                dtype=np.float32,
            )

            progress[:, full_year] = annual_progress[:, None]

        # --------------------------------------------------------------
        # Linear interpolation
        # --------------------------------------------------------------

        values = (
            start_value
            + progress * (end_value - start_value)
        ).astype(np.float32)

        values[~stage_mask] = 0.0

        return values

    def build_Kc_from_crop(self, crop: Crop) -> xr.DataArray:
        """
        Build the daily crop coefficient (Kc) matrix from the 36-dekad
        phenology state matrix.

        The following interpolation is applied:

            inactive:
                Kc = inactive_kc

            growing:
                Kc_initial -> Kc_max

            maximum:
                Kc = Kc_max

            senescence:
                Kc_max -> Kc_sen

        Parameters
        ----------
        crop : Crop
            Crop object containing:

                crop.kc_initial
                crop.kc_max
                crop.kc_sen

        Returns
        -------
        xr.DataArray
            Daily Kc matrix with dimensions:

                time = 365
                row
                col
        """

        if crop is None:
            raise ValueError("crop cannot be None")

        self.crop = crop

        # --------------------------------------------------------------
        # Convert 36 dekads -> 365 days
        # --------------------------------------------------------------

        daily_state = self._state_to_daily()

        # --------------------------------------------------------------
        # Start with inactive Kc everywhere
        # --------------------------------------------------------------

        kc = np.full(
            daily_state.shape,
            self.inactive_kc,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Growing phase
        #
        # Kc increases linearly:
        #
        #     kc_initial -> kc_max
        # --------------------------------------------------------------

        growing_mask = (
            daily_state == PHENOLOGY_GROWING
        )

        growing_kc = self._interpolate_stage(
            stage_mask=growing_mask,
            start_value=crop.kc_initial,
            end_value=crop.kc_max,
        )

        kc[growing_mask] = growing_kc[growing_mask]

        # --------------------------------------------------------------
        # Maximum phase
        #
        # Constant Kc_max
        # --------------------------------------------------------------

        maximum_mask = (
            daily_state == PHENOLOGY_MAXIMUM
        )

        kc[maximum_mask] = np.float32(
            crop.kc_max
        )

        # --------------------------------------------------------------
        # Senescence
        #
        # Kc decreases linearly:
        #
        #     kc_max -> kc_sen
        # --------------------------------------------------------------

        senescence_mask = (
            daily_state == PHENOLOGY_SENESCENCE
        )

        senescence_kc = self._interpolate_stage(
            stage_mask=senescence_mask,
            start_value=crop.kc_max,
            end_value=crop.kc_sen,
        )

        kc[senescence_mask] = (
            senescence_kc[senescence_mask]
        )

        # --------------------------------------------------------------
        # Spatial validity mask
        #
        # For now invalid pixels retain inactive_kc.
        # If you prefer NaN, use:
        #
        # kc[:, ~self.valid_mask.astype(bool)] = np.nan
        # --------------------------------------------------------------

        valid = self.valid_mask.astype(bool)

        kc[:, ~valid] = self.inactive_kc

        # --------------------------------------------------------------
        # Store as xarray.DataArray
        # --------------------------------------------------------------

        self.Kc_matrix = xr.DataArray(
            kc,
            dims=("time", "row", "col"),
            coords={
                "time": np.arange(1, 367),
                "row": self.state.row.values,
                "col": self.state.col.values,
            },
            name="Kc",
            attrs={
                "long_name": "daily crop coefficient",
                "crop": getattr(crop, "name", None),
                "kc_initial": float(crop.kc_initial),
                "kc_max": float(crop.kc_max),
                "kc_sen": float(crop.kc_sen),
                "inactive_kc": float(self.inactive_kc),
            },
        )

        return self.Kc_matrix
