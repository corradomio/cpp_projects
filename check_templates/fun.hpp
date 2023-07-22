//
// Created by Corrado Mio on 21/07/2023.
//

#ifndef CHECK_TEMPLATES_FUN_HPP
#define CHECK_TEMPLATES_FUN_HPP

#include <cstdint>
#include <string>
#include <memory>
#include <functional>
#include <wstp.h>

namespace ErrorName {

    // Original LibraryLink error codes:
    extern const std::string VersionError;		 ///< same as LIBRARY_VERSION_ERROR
    extern const std::string FunctionError;		 ///< same as LIBRARY_FUNCTION_ERROR
    extern const std::string MemoryError;		 ///< same as LIBRARY_MEMORY_ERROR
    extern const std::string NumericalError;	 ///< same as LIBRARY_NUMERICAL_ERROR
    extern const std::string DimensionsError;	 ///< same as LIBRARY_DIMENSIONS_ERROR
    extern const std::string RankError;			 ///< same as LIBRARY_RANK_ERROR
    extern const std::string TypeError;			 ///< same as LIBRARY_TYPE_ERROR
    extern const std::string NoError;			 ///< same as LIBRARY_NO_ERROR

    // LibraryData errors
    extern const std::string LibDataError;	  ///< WolframLibraryData is not set

    // MArgument errors:
    extern const std::string MArgumentIndexError;			///< wrong argument index
    extern const std::string MArgumentNumericArrayError;	///< error involving NumericArray argument
    extern const std::string MArgumentTensorError;			///< error involving Tensor argument
    extern const std::string MArgumentImageError;			///< error involving Image argument

    // ErrorManager errors:
    extern const std::string ErrorManagerThrowIdError;		 ///< trying to throw exception with non-existent id
    extern const std::string ErrorManagerThrowNameError;	 ///< trying to throw exception with non-existent name
    extern const std::string ErrorManagerCreateNameError;	 ///< trying to register exception with already existing name

    // NumericArray errors:
    extern const std::string NumericArrayNewError;			 ///< creating new NumericArray failed
    extern const std::string NumericArrayCloneError;		 ///< NumericArray cloning failed
    extern const std::string NumericArrayTypeError;			 ///< NumericArray type mismatch
    extern const std::string NumericArraySizeError;			 ///< wrong assumption about NumericArray size
    extern const std::string NumericArrayIndexError;		 ///< trying to access non-existing element
    extern const std::string NumericArrayConversionError;	 ///< conversion from NumericArray of different type failed

    // MTensor errors:
    extern const std::string TensorNewError;	  ///< creating new MTensor failed
    extern const std::string TensorCloneError;	  ///< MTensor cloning failed
    extern const std::string TensorTypeError;	  ///< Tensor type mismatch
    extern const std::string TensorSizeError;	  ///< wrong assumption about Tensor size
    extern const std::string TensorIndexError;	  ///< trying to access non-existing element

    // SparseArray errors:
    extern const std::string SparseArrayCloneError;					///< could not clone MSparseArray
    extern const std::string SparseArrayTypeError;					///< SparseArray type mismatch
    extern const std::string SparseArrayFromPositionsError;			///< could not create MSparseArray from explicit positions
    extern const std::string SparseArrayFromTensorError;			///< could not create MSparseArray from MTensor
    extern const std::string SparseArrayImplicitValueResetError;	///< could not reset the implicit value of MSparseArray
    extern const std::string SparseArrayImplicitValueError;			///< could not read implicit value from MSparseArray
    extern const std::string SparseArrayExplicitValuesError;		///< could not read explicit values from MSparseArray
    extern const std::string SparseArrayRowPointersError;			///< could not read row pointers from MSparseArray
    extern const std::string SparseArrayColumnIndicesError;			///< could not read column indices from MSparseArray
    extern const std::string SparseArrayExplicitPositionsError;		///< could not read explicit positions from MSparseArray
    extern const std::string SparseArrayToTensorError;				///< could not dump MSparseArray to MTensor

    // MImage errors:
    extern const std::string ImageNewError;		 ///< creating new MImage failed
    extern const std::string ImageCloneError;	 ///< MImage cloning failed
    extern const std::string ImageTypeError;	 ///< Image type mismatch
    extern const std::string ImageSizeError;	 ///< wrong assumption about Image size
    extern const std::string ImageIndexError;	 ///< trying to access non-existing element

    // General container errors:
    extern const std::string CreateFromNullError;		   ///< attempting to create a generic container from nullptr
    extern const std::string MArrayElementIndexError;	   ///< attempting to access MArray element at invalid index
    extern const std::string MArrayDimensionIndexError;	   ///< attempting to access MArray dimension at invalid index

    // WSTP errors:
    extern const std::string WSNullWSLinkError;			   ///< Trying to create WSStream with NULL WSLINK
    extern const std::string WSTestHeadError;			   ///< WSTestHead failed (wrong head or number of arguments)
    extern const std::string WSPutSymbolError;			   ///< WSPutSymbol failed
    extern const std::string WSPutFunctionError;		   ///< WSPutFunction failed
    extern const std::string WSTestSymbolError;			   ///< WSTestSymbol failed (different symbol on the link than expected)
    extern const std::string WSWrongSymbolForBool;		   ///< Tried to read something else than "True" or "False" as boolean
    extern const std::string WSGetListError;			   ///< Could not get list from WSTP
    extern const std::string WSGetScalarError;			   ///< Could not get scalar from WSTP
    extern const std::string WSGetStringError;			   ///< Could not get string from WSTP
    extern const std::string WSGetArrayError;			   ///< Could not get array from WSTP
    extern const std::string WSPutListError;			   ///< Could not send list via WSTP
    extern const std::string WSPutScalarError;			   ///< Could not send scalar via WSTP
    extern const std::string WSPutStringError;			   ///< Could not send string via WSTP
    extern const std::string WSPutArrayError;			   ///< Could not send array via WSTP
    extern const std::string WSGetSymbolError;			   ///< WSGetSymbol failed
    extern const std::string WSGetFunctionError;		   ///< WSGetFunction failed
    extern const std::string WSPacketHandleError;		   ///< One of the packet handling functions failed
    extern const std::string WSFlowControlError;		   ///< One of the flow control functions failed
    extern const std::string WSTransferToLoopbackError;	   ///< Something went wrong when transferring expressions from loopback link
    extern const std::string WSCreateLoopbackError;		   ///< Could not create a new loopback link
    extern const std::string WSLoopbackStackSizeError;	   ///< Loopback stack size too small to perform desired action

    // DataList errors:
    extern const std::string DLNullRawNode;			 ///< DataStoreNode passed to Node wrapper was null
    extern const std::string DLInvalidNodeType;		 ///< DataStoreNode passed to Node wrapper carries data of invalid type
    extern const std::string DLGetNodeDataError;	 ///< DataStoreNode_getData failed
    extern const std::string DLSharedDataStore;	 	 ///< Trying to create a Shared DataStore. DataStore can only be passed as Automatic or Manual.
    extern const std::string DLPushBackTypeError;	 ///< Element to be added to the DataList has incorrect type

    // MArgument errors:
    extern const std::string ArgumentCreateNull;		  ///< Trying to create PrimitiveWrapper object from nullptr
    extern const std::string ArgumentAddNodeMArgument;	  ///< Trying to add DataStore Node of type MArgument (aka MType_Undef)

    // ProgressMonitor errors:
    extern const std::string Aborted;	 ///< Computation aborted by the user

    // ManagedExpression errors:
    extern const std::string ManagedExprInvalidID;	  ///< Given number is not an ID of any existing managed expression
    extern const std::string MLEDynamicTypeError;	  ///< Invalid dynamic type requested for a Managed Library Expression
    extern const std::string MLENullInstance; 		  ///< Missing managed object for a valid ID

    // FileUtilities errors:
    extern const std::string PathNotValidated;		///< Given file path could not be validated under desired open mode
    extern const std::string InvalidOpenMode;		///< Specified open mode is invalid
    extern const std::string OpenFileFailed;		///< Could not open file
}  // namespace ErrorName

enum class Encoding : std::uint8_t {
    Undefined,	  //!< Undefined, can be used to denote that certain function is not supposed to deal with strings
    Native,		  //!< Use WSGetString for reading and WSPutString for writing strings
    Byte,		  //!< Use WSGetByteString for reading and WSPutByteString for writing strings
    UTF8,		  //!< Use WSGetUTF8String for reading and WSPutUTF8String for writing strings
    UTF16,		  //!< Use WSGetUTF16String for reading and WSPutUTF16String for writing strings
    UCS2,		  //!< Use WSGetUCS2String for reading and WSPutUCS2String for writing strings
    UTF32		  //!< Use WSGetUTF32String for reading and WSPutUTF32String for writing strings
};

namespace Detail {
    /**
     * @brief 		Checks if WSTP operation was successful and throws appropriate exception otherwise
     * @param[in] 	m - low-level object of type WSLINK received from LibraryLink
     * @param[in] 	statusOk - status code return from a WSTP function
     * @param[in] 	errorName - what error name to put in the exception if WSTP function failed
     * @param[in] 	debugInfo - additional info to be attached to the exception
     */
    void checkError(WSLINK m, int statusOk, const std::string& errorName, const std::string& debugInfo = "");

    /**
     * @brief	Simple wrapper over ErrorManager::throwException used to break dependency cycle between WSStream and ErrorManager.
     * @param 	errorName - what error name to put in the exception
     * @param 	debugInfo - additional info to be attached to the exception
     */
    [[noreturn]] void throwLLUException(const std::string& errorName, const std::string& debugInfo = "");

    /**
     * @brief	Returns a new loopback link using WSLinkEnvironment(m) as WSENV
     * @param 	m - valid WSLINK
     * @return 	a brand new Loopback Link
     */
    WSLINK getNewLoopback(WSLINK m);

    /**
     * @brief	Get the number of expressions stored in the loopback link
     * @param	lpbckLink - a reference to the loopback link, after expressions are counted this argument will be assigned a different WSLINK
     * @return	a number of expression stored in the loopback link
     */
    int countExpressionsInLoopbackLink(WSLINK& lpbckLink);
}	 // namespace Detail


template<Encoding E>
struct CharTypeStruct {
    static_assert(E != Encoding::Undefined, "Trying to deduce character type for undefined encoding");
    /// static_assert will trigger compilation error, we add a dummy type to prevent further compiler errors
    using type = char;
};

/**
 * Specializations of CharTypeStruct, encoding E has assigned type T iff WSPutEString takes const T* as second parameter
 * @cond
 */
template<>
struct CharTypeStruct<Encoding::Native> {
    using type = char;
};
template<>
struct CharTypeStruct<Encoding::Byte> {
    using type = unsigned char;
};
template<>
struct CharTypeStruct<Encoding::UTF8> {
    using type = unsigned char;
};
template<>
struct CharTypeStruct<Encoding::UTF16> {
    using type = unsigned short;
};
template<>
struct CharTypeStruct<Encoding::UCS2> {
    using type = unsigned short;
};
template<>
struct CharTypeStruct<Encoding::UTF32> {
    using type = unsigned int;
};

template<Encoding E>
using CharType = typename CharTypeStruct<E>::type;

template<Encoding E, typename T>
inline constexpr bool CharacterTypesCompatible = (sizeof(T) == sizeof(CharType<E>));


template<Encoding E>
struct ReleaseString;

/// StringData with Encoding \p E is a unique_ptr to an array of E-encoded characters
/// It allows you to take ownership of raw string data from WSTP without making extra copies.
template<Encoding E>
using StringData = std::unique_ptr<const CharType<E>[], ReleaseString<E>>;

/// GetStringFuncT is a type of WSTP function that reads string from a link, e.g. WSGetByteString
template<typename T>
using GetStringFuncT = std::function<int(WSLINK, const T**, int*, int*)>;

/// PutStringFuncT is a type of WSTP function that sends string data to a link, e.g. WSPutByteString
template<typename T>
using PutStringFuncT = std::function<int(WSLINK, const T*, int)>;

/// ReleaseStringFuncT is a type of WSTP function to release string data allocated by WSTP, e.g. WSReleaseByteString
template<typename T>
using ReleaseStringFuncT = std::function<void(WSLINK, const T*, int)>;


template<Encoding E>
struct String {

    using CharT = CharType<E>;

    static GetStringFuncT<CharT> Get;
    static PutStringFuncT<CharT> Put;
    static ReleaseStringFuncT<CharT> Release;

    static const std::string GetFName;
    static const std::string PutFName;

    template<typename T>
    static void put(WSLINK m, const T* string, int len) {
        static_assert(CharacterTypesCompatible<E, T>, "Character type does not match the encoding in WS::String<E>::put");
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): sorry :(
        auto* expectedStr = reinterpret_cast<const CharT*>(string);
        Detail::checkError(m, Put(m, expectedStr, len), ErrorName::WSPutStringError, PutFName);
    }

    static StringData<E> get(WSLINK m) {
        const CharT* rawResult {};
        int bytes {};
        int characters {};
        Detail::checkError(m, Get(m, &rawResult, &bytes, &characters), ErrorName::WSGetStringError, GetFName);
        return {rawResult, ReleaseString<E> {m, bytes, characters}};
    }

    template<typename T>
    static std::basic_string<T> getString(WSLINK m) {
        static_assert(CharacterTypesCompatible<E, T>, "Character type does not match the encoding in WS::String<E>::getString");
        using StringType = std::basic_string<T>;

        auto strData {get(m)};

        auto bytes = strData.get_deleter().getLength();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): sorry :(
        auto* expectedData = reinterpret_cast<const T*>(strData.get());
        auto strlen = static_cast<typename StringType::size_type>(bytes);

        return (bytes < 0 ? StringType {expectedData} : StringType {expectedData, strlen});
    }
};


int fun1(int x);
int fun2(int x);

#endif //CHECK_TEMPLATES_FUN_HPP
