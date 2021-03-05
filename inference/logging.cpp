//Global definition of logging object
#include "include/logging.h"


namespace trtInference {

static Logger gLogger{Logger::Severity::kINFO};
static LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
static LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
static LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
static LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
static LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)}; 

}



