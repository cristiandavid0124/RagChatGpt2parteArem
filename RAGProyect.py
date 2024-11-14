import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

web_paths = [
    "https://medium.com/psicolog%C3%ADa-del-lenguaje-ugr/por-qu%C3%A9-importan-los-videojuegos-c6124536bd64",
    "https://medium.com/@reneblanco/los-10-mejores-videojuegos-6d3e07e6ceb7",
    "https://medium.com/los-inrockuptibles/por-que-los-videojuegos-estan-cambiando-el-deporte-4cf7fbaafe09",
    "https://medium.com/cachivache-media/comunidades-videojuegos-cuba-f922cc1cfaf1",
]


loader = WebBaseLoader(
    web_paths=web_paths,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "du", "ag", "dv", "bf", "ak", "am", "an", "ao", "ap", "aq", "ar", "as", "at", "dw", "dt", "ab", "dx", "dy", "dz", "ea", "eb", "ec", "ed", "ee", "ef", "eg", "eh", "ei", "ej", "ek", "el", "em", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ew", "ex", "ey", "ez", "fa", "fb", "fc", "fd", "bm", "fe", "ff", "ax", "af", "ah", "ai", "aj", "al", "ac", "ae", "au", "av", "aw", "ay", "az", "ba", "bb", "bc", "bd", "bn", "bo", "be", "bg", "bh", "bi", "bj", "bk", "bl", "fg", "fh", "fi", "fj", "fk", "fl", "fm", "fn", "fo", "fp", "fq", "by", "bz", "ca", "cx", "fr", "fs", "ft", "ra", "sp", "sq", "sr", "ss", "st", "su", "gm", "sv", "sw", "sx", "sy", "sz", "ta", "tb", "tc", "lc", "td", "te", "tf", "tg", "th", "ti", "tj", "tk", "tl", "fu", "fv", "fw", "fx", "fy", "cb", "ci", "fz", "ga", "gb", "gc", "gi", "gj", "gk", "gl", "mh", "mi", "to", "speechify-ignore", "qh", "lo", "act", "qo", "tt", "tu", "gn", "go", "gp", "gq", "gr", "pw-post-title", "gs", "gt", "gu", "gv", "gw", "gx", "gy", "gz", "ha", "hb", "hc", "hd", "he", "hf", "hg", "hh", "hi", "hj", "hk", "hl", "hm", "hn", "ho", "hp", "hq", "hr", "hs", "ht", "hu", "tp", "tq", "cp", "hv", "hw", "hx", "hy", "hz", "ia", "ib", "ic", "id", "ie", "dd", "de", "if", "ig", "ih", "ii", "ij", "ik", "il", "im", "in", "io", "ip", "iq", "ir", "is", "cn", "it", "iu", "iv", "iw", "ix", "iy", "iz", "ja", "jb", "jc", "jd", "je", "jf", "jg", "jh", "ji", "jj", "jk", "jl", "jm", "jn", "kd", "ke", "kf", "pw-multi-vote-icon", "kg", "kh", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kp", "kq", "kr", "pw-multi-vote-count", "ks", "kt", "ku", "kv", "kw", "kx", "ky", "tr", "lh", "rf", "rg", "ld", "le", "lf", "lb", "pw-responses-count", "la", "jo", "jp", "jq", "jr", "js", "jt", "ju", "jv", "jw", "jx", "jy", "jz", "ka", "kb", "kc", "lg", "li", "lj", "lk", "ll", "lm", "ln", "lp", "lq", "lr", "ls", "lt", "lu", "lv", "lw", "lx", "ly", "lz", "ma", "mb", "mc", "md", "me", "mf", "mg", "mk", "ml", "mm", "mn", "mo", "mp", "paragraph-image", "mq", "mr", "ms", "mt", "mj", "mu", "pw-post-body-paragraph", "mv", "mw", "mx", "my", "mz", "na", "nb", "nc", "nd", "ne", "nf", "ng", "nh", "ni", "nj", "nk", "nl", "nm", "nn", "no", "np", "nq", "nr", "ns", "nt", "nu", "nv", "nw", "nx", "ny", "nz", "oa", "ob", "oc", "od", "oe", "of", "og", "oh", "oi", "oj", "ok", "ol", "om", "on", "oo", "op", "oq", "or", "os", "ot", "ou", "ov", "ow", "ox", "oy", "oz", "pa", "pb", "pc", "pd", "pe", "pf", "pg", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pp", "pq", "pr", "ps", "pt", "pu", "pv", "pw", "px", "py", "pz", "qa", "qb", "qc", "qd", "qe", "qf", "abl", "acu", "ade", "acw", "acx", "acy", "acz", "qg", "qi", "qj", "qk", "ql", "ge", "qm", "qn", "qp", "qq", "qr", "qs", "qt", "qu", "qv", "qw", "qx", "qy", "qz", "rb", "rc", "rd", "re", "bq", "rh", "ri", "rj", "rk", "rl", "bx", "cl", "rm", "ada", "adb", "rp", "adc", "add", "tn", "rt", "ru", "pw-author-name", "se", "sf", "sg", "sh", "pw-follower-count", "si", "sj", "sk", "tv", "tw", "tx", "ty", "tz", "ua", "ub", "uc", "ud", "ue", "uf", "ug", "uh", "ui", "uj", "uk", "ul", "um", "un", "uo", "up", "uq", "ur", "us", "ut", "uu", "uv", "uw", "ux", "uy", "uz", "va", "vb", "vc", "vd", "ve", "vf", "vg", "vh", "vi", "vj", "vk", "vl", "vm", "vn", "vo", "vp", "vq", "vr", "vs", "vt", "vu", "vv", "vw", "vx", "vy", "vz", "wa", "wc", "wd", "we", "wf", "wg", "wh", "wi", "wj", "wk", "wb", "co", "wl", "wm", "wn", "wo", "wp", "wq", "wr", "ws", "wt", "wu", "wv", "ww", "wx", "wy", "wz", "xa", "xb", "xc", "xd", "xe", "xf", "xg", "xh", "xi", "xj", "xk", "xl", "xm", "xn", "xo", "xp", "xq", "xr", "xs", "xt", "xu", "xv", "xw", "xx", "xy", "xz", "ya", "yb", "yc", "yd", "ye", "yh", "yf", "dh", "yg", "dj", "yi", "yj", "yk", "yl", "dk", "dl", "ym", "yn", "yo", "yp", "yq", "yr", "ys", "yt", "yu", "yv", "yw", "yx", "yy", "yz", "za", "zb", "zc", "zd", "ze", "zf", "zg", "zh", "zi", "zj", "zk", "zl", "zm", "zn", "zo", "zp", "zq", "zw", "zx", "zs", "zt", "zu", "zv", "zr", "zy", "zz", "aba", "abb", "abc", "abd", "abe", "abf", "abg", "abh", "abi", "sl", "sm", "sn", "so")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_rag_response(question: str) -> str:
    return rag_chain.invoke(question)