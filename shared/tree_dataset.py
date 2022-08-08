import h5py
from typing import Union, Callable, List, Dict
import unyt

class RegistrationDoesNotMatchError(Exception):
    """
    Raised when our registration function does not match the current
    variable name passed to it.
    """
    def __init__(self, message=""):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def registration_catalogue(field_path: str) -> (str, str):
    """
    Registers all quantities related to halo ids (those beginning or ending with ID).
    """
    if not field_path[:2] in ["ID", "Pr", "Nu"]:
        raise RegistrationDoesNotMatchError

    if field_path == "ID":
        full_name = "Halo ID"
    elif field_path == "Progenitors":
        full_name = "Progenitors"
    elif field_path == "NumProgen":
        full_name = "NumProgen"
    elif field_path == "ProgenOffsets":
        full_name = "ProgenOffsets"
    else:
        full_name = "Generic ID"
    return full_name, field_path

# This must be placed at the bottom of the file so that we
# have defined all functions before getting to it.
# This dictionary will be turned into sets of datasets that
# contain the results of the registraiton functions.
global_registration_functions = {
    k: globals()[f"registration_{k}"]
    for k in [
        "catalogue",
    ]
}

class TREEFieldMetadata(object):
    """
    Metadata for a merger tree field. Pass it a field path and a filename,
    and it will:
    + Use the registration functions to find the correct units and
      "fancy name".
    + Assign a proper snake_case_name to the dataset.
    """

    # Forward declarations for the field.

    # Is this a valid field?
    valid: bool = False
    # The fancy name of this field for plotting, provided by registration function
    name: str = ""
    # The snake case name of this field for use accessing the object.
    snake_case: str = ""
    # The registartion function that matched with this field
    corresponding_registration_function_name: Union[str, None]
    corresponding_registration_function: Union[Callable, None]

    def __init__(
        self,
        filename,
        path: str,
        registration_functions: Dict[str, Callable],
    ):
        """
        I take in:
        + filename of the velociraptor properties file
        + path of the field you wish me to look at
        + registration_functions a list of callables with the registration
          function signature (see registration.py or the documentation)
        + units, a pointer or copy of the unit system associated with this file.
        """

        # Filename not currently used but may be required later on if
        # actual field metadata is included in the velociraptor properties files
        self.filename = filename
        self.path = path
        self.registration_functions = registration_functions

        self.register_field_properties()

        return

    def register_field_properties(self):
        """
        Registers the field properties using the registration functions.
        """
        for reg_name, reg in self.registration_functions.items():
            try:
                self.name, self.snake_case = reg(field_path=self.path)
                self.valid = True
                self.corresponding_registration_function = reg
                self.corresponding_registration_function_name = reg_name
            except RegistrationDoesNotMatchError:
                continue

        return

def generate_getter(filename, name: str, field: str, full_name: str):
    """
    Generates a function that:
    a) If self._`name` exists, return it
    b) If not, open `filename`
    c) Reads filename[`field`]
    d) Set self._`name`
    e) Return self._`name`.
    Takes:
    + filename, the filename of the hdf5 file
    + name, the snake_case name of the property
    + field, the field in the hdf5 file corresponding to this property
    + full_name, the fancy printing name for this quantity (registered to array.name)
    + unit, the unyt unit corresponding to this value
    """

    def getter(self):
        current_value = getattr(self, f"_{name}")

        if current_value is not None:
            return current_value
        else:
            with h5py.File(filename, "r") as handle:
                try:
                    setattr(self, f"_{name}", unyt.unyt_array(handle[field][...]))
                    getattr(self, f"_{name}").name = full_name
                    getattr(self, f"_{name}").file = filename
                except KeyError:
                    print(f"Could not read {field}")
                    return None

        return getattr(self, f"_{name}")

    return getter


def generate_setter(name: str):
    """
    Generates a function that sets self._name to the value that is passed to it.
    """

    def setter(self, value):
        setattr(self, f"_{name}", value)

        return

    return setter


def generate_deleter(name: str):
    """
    Generates a function that destroys self._name (sets it back to None).
    """

    def deleter(self):
        current_value = getattr(self, f"_{name}")
        del current_value
        setattr(self, f"_{name}", None)

        return
    return deleter

def generate_sub_catalogue(
    filename,
    registration_name: str,
    registration_function: Callable,
    field_metadata: List[TREEFieldMetadata],
):
    """
    Generates a sub-catalogue object with the correct properties set.
    This is required as we can add properties to a class, but _not_
    to an object dynamically.
    So, here, we initialise the metadata, create a _copy_ of the
    __VelociraptorSubCatlaogue class, and then add all of our properties
    to that _class_ before instantiating it with the metadata.
    """

    # This creates a _copy_ of the _class_, not object.
    this_sub_catalogue_bases = (
        __TreeSubCatalogue,
        object,
    )
    this_sub_catalogue_dict = {}

    valid_sub_paths = []

    for metadata in field_metadata:
        valid_sub_paths.append(metadata.snake_case)

        this_sub_catalogue_dict[metadata.snake_case] = property(
            generate_getter(
                filename,
                metadata.snake_case,
                metadata.path,
                metadata.name,
            ),
            generate_setter(metadata.snake_case),
            generate_deleter(metadata.snake_case),
        )

        this_sub_catalogue_dict[f"_{metadata.snake_case}"] = None

    ThisSubCatalogue = type(
        f"Dynamic_{registration_name}_TreeCatalogue",
        this_sub_catalogue_bases,
        this_sub_catalogue_dict,
    )

    # Finally, we can actually create an instance of our new class.
    catalogue = ThisSubCatalogue(filename=filename)
    catalogue.valid_sub_paths = valid_sub_paths

    return catalogue

class __TreeSubCatalogue(object):
    """
    A velociraptor mini-catalogue, containing the only the information from one
    registration function. This allows us to separate the top-level variables
    into more manageable chunks.
    Do not directly instantiate this class, you should use generate_sub_catalogue.
    This is called in VelociraptorCatalogue.
    """

    # The valid paths contained within
    valid_sub_paths: List[str]

    def __init__(self, filename):
        self.filename = filename

        return

    def __str__(self):
        return f"Contains the following fields: {', '.join(self.valid_sub_paths)}"

    def __repr__(self):
        return str(self)

class TreeCatalogue(object):
    """
    A merger tree file dataset, containing all the information.
    """

    # Top-level definitions for autocomplete
    registration_functions: Union[List[Callable], None]

    def __init__(
        self,
        filename: str
    ):
        """
        Initialise the velociraptor catalogue with all of the available
        datasets. This class should never be instantiated manually and should
        always be handled through the generate_catalogue function.
        Parameters
        ----------
        filename: str
            File path to the merger tree file that you wish to open.

        """
        self.filename = filename
        self.registration_functions = global_registration_functions
        self.__create_sub_catalogues()
        return

    def __str__(self):
        """
        Prints out some more useful information, rather than just
        the memory location.
        """

        return (
            f"Merger tree dataset at {self.filename}. "
            "Contains the following field collections: "
            f"{', '.join(self.valid_field_metadata.keys())}"
        )

    def __repr__(self):
        return str(self)

    def __create_sub_catalogues(self):
        """
        Creates the sub-catalogues by instantiating many different versions
        of the __VelociraptorSubCatalogue. Each sub-catalogue corresponds to
        the output of a single registration function.
        """

        # First load all field names from the HDF5 file so that they can be parsed.

        with h5py.File(self.filename, "r") as handle:
            field_paths = list(handle.keys())

        # Now build metadata:
        self.valid_field_metadata = {
            reg: [] for reg in self.registration_functions.keys()
        }
        self.invalid_field_paths = []

        for path in field_paths:
            metadata = TREEFieldMetadata(
                self.filename, path, self.registration_functions
            )

            if metadata.valid:
                self.valid_field_metadata[
                    metadata.corresponding_registration_function_name
                ].append(metadata)
            else:
                self.invalid_field_paths.append(path)

        # For each registration function, we create a dynamic sub-class that
        # contains only that information - otherwise the namespace of the
        # VelociraptorCatalogue is way too crowded.
        for attribute_name, field_metadata in self.valid_field_metadata.items():
            setattr(
                self,
                attribute_name,
                generate_sub_catalogue(
                    filename=self.filename,
                    registration_name=attribute_name,  # This ensures each class has a unique name
                    registration_function=self.registration_functions[attribute_name],
                    field_metadata=field_metadata,
                ),
            )

        return
