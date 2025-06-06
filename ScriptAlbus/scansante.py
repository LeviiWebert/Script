import random
from time import sleep
from types import TracebackType
from typing import Any, Dict, List

import click
import requests
import MySQLdb as mysql_db
import MySQLdb.cursors as cursors
from bs4 import BeautifulSoup
from MySQLdb.connections import Connection

import re # Ajout GI 20250408

class DatabaseException(Exception):
    pass


class DatabaseConnectionError(DatabaseException):
    pass


class DatabaseQueryExecError(DatabaseException):
    pass


def prep_for_casting(value: str) -> str:
    """Prepare the value for casting by removing unwanted characters."""
    value = value.replace("%", "").replace(",", ".").replace(" ", "")
    return value.strip()


def cast_int(value):
    try:
        return int(prep_for_casting(value))
    except ValueError:
        return value


def cast_float(value):
    try:
        return float(prep_for_casting(value))
    except ValueError:
        return value


def cast_percentage(value):
    try:
        return float(prep_for_casting(value)) / 100
    except ValueError:
        return value


class OutputHandler:
    def __enter__(self) -> "MysqlHandler":
        raise NotImplementedError("OutputHandler is an abstract class and should not be instantiated directly.")

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        raise NotImplementedError("OutputHandler is an abstract class and should not be instantiated directly.")

    def write(self, data: Dict[str, str | int | float]) -> None:
        raise NotImplementedError("OutputHandler is an abstract class and should not be instantiated directly.")


class MysqlHandler(OutputHandler):
    COLMAPPING = (
        ("annee", "annee"),
        ("finess", "finess"),
        ("Code", "code"),
        ("Libellé", "label"),
        ("Effectif", "headcount"),
        ("Durée moyennede séjour", "avg_journey"),
        ("Age moyen", "avg_age"),
        ("Sexe ratio(% homme)", "p_male"),
        ("% décès", "p_death"),
    )

    def __init__(self, db_url: str) -> None:
        self.is_connected: bool = False
        self.conn: Any = None
        self.split_url(db_url)

    def split_url(self, url: str) -> Dict[str, str]:
        if not url.startswith("mysql://"):
            raise ValueError("Invalid URL format. Expected format: mysql://user:password@host:port/database")
        url = url[8:]
        self.user, rest = url.split(":", 1)
        self.password, rest = rest.split("@", 1)
        self.host, rest = rest.split(":", 1)
        self.port, self.database = rest.split("/", 1)

    def connect(self) -> Connection:
        try:
            if not self.is_connected:
                self.conn = mysql_db.connect(
                    host=self.host,
                    port=int(self.port),
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    cursorclass=cursors.DictCursor,
                )
            return self.conn
        except mysql_db.Error as exc:
            raise DatabaseConnectionError(
                f"Error connecting to db with {[self.user, self.host, self.port, self.database]}: {exc}"
            ) from exc

    def __enter__(self) -> "MysqlHandler":
        self.connect()
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        if self.is_connected:
            self.conn.close()

    def execute(self, sql: str, params: List | None = None) -> List[Dict | List]:
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()
        except mysql_db.Error as exc:
            raise DatabaseQueryExecError(f"Error executing SQL query: {sql} with {params}") from exc
        finally:
            cursor.close()

    def write(self, data: Dict[str, str | int | float]) -> None:
        # order correctly parameters
        params = [data[key_in] for key_in, key_out in self.COLMAPPING]
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                (
                    "INSERT INTO scansante "
                    "(annee, finess, code, label, headcount, avg_journey, avg_age, p_male, p_death) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
                ),
                params,
            )
            conn.commit()
        except mysql_db.Error as exc:
            raise DatabaseQueryExecError(f"Error writing to database: {exc}") from exc
        finally:
            cursor.close()

    def create_table(self) -> None:
        self.execute("DROP TABLE IF EXISTS scansante;")
        self.execute(
            "CREATE TABLE IF NOT EXISTS scansante ("
            "id INT AUTO_INCREMENT PRIMARY KEY, "
            "annee INT unsigned, "
            "finess VARCHAR(10), "
            "code VARCHAR(10), "
            "label VARCHAR(255), "
            "headcount INT, "
            "avg_journey FLOAT, "
            "avg_age FLOAT, "
            "p_male FLOAT, "
            "p_death FLOAT"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;"
        )
    # Debut de Modif par GI 20250408        
    def found_finess(self, finess : str, annee: str) -> bool:
        rep=self.execute(f"select count(*) as nb from scansante where finess = '{finess}' and annee={annee};")
        if rep[0]['nb'] > 0:
            return True
        return False
    # Fin de Modif par GI 20250408        

class FileHandler(OutputHandler):
    def __init__(self, file_path: click.Path) -> None:
        self.file_path = file_path
        self.file_pointer = None

    def __enter__(self) -> "MysqlHandler":
        self.file_pointer = open(self.file_path, "w")
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        self.file_pointer.close()

    def write(self, data: List[str]) -> None:
        # cast to str and encapsulate data within double quotes
        data = [f'"{str(_)}"' for _ in data.values()]
        self.file_pointer.write(";".join(data) + "\n")


class Scraper:
    # Modif par GI 20250408
    URLMASK = (
        "https://www.scansante.fr/applications/casemix_ghm_cmd/submit?snatnav=&typrgp=etab&annee={annee}"
        "&type=ghm&base=0&typreg=noreg&noreg=99&finess={finess}&editable_length=10"
    )
    CASTMAPPING = (
        ("Effectif", cast_int),
        ("Durée moyennede séjour", cast_float),
        ("Age moyen", cast_float),
        ("Sexe ratio(% homme)", cast_percentage),
        ("% décès", cast_percentage),
    )

    def get_data(self, finess: str, annee: str) -> List[Dict[str, str | int | float]]:   # Modif par GI 20250408
        """Get data from the web page."""
        # Debut de Modif par GI 20250408
        #content = self._fetch_web_page_content(finess) 
        content = self._fetch_web_page_content(finess, annee)
        if not content:
            return False
        # Fin de Modif par GI 20250408
        data = self._parse_web_page_content(content, finess)  # Modif par GI 20250408
        #data = [{"annee": annee} | _ for _ in data]           # Modif par GI 20250408
        return data

    def _fetch_web_page_content(self, finess: str, annee: str) -> str:  # Modif par GI 20250408
        """Use requests to fetch the web page content."""
        url = self.URLMASK.format(finess=finess, annee=annee)  # Modif par GI 20250408
        response = requests.get(url)
        # Debut de Modif par GI 20250408
        #response.raise_for_status()
        if response.status_code >= 500:
            click.echo(f" ERREUR HTTP {response.status_code} pour requete sur finess {finess} et année {annee}.")
            return False
        elif response.status_code != 200:
            response.raise_for_status()
        # Fin Modif par GI 20250408            

        return response.text

    def _parse_web_page_content(self, content: str, finess: str) -> List[Dict[str, str | int | float]]:   # Modif par GI 20250408
        """Parse the web page content using BeautifulSoup.

        1. find the correct table using summary attribute
        2. get the headers from the table to build pretty dict
        3. get the data from the table
        4. cast columns to get correct types
        5. return the data
        """
        try:
            soup = BeautifulSoup(content, "html.parser")
        
            # Ajout GI 20250408
            erreur = soup.find("div", class_="error")
            if erreur:
                if not erreur.text:
                    message=erreur.string
                    raise ValueError(f"Erreur retournée pour le finess {finess} par le site: {message}")
                else:
                    raise ValueError(f"Erreur retournée pour le finess {finess} par le site: {erreur.text}")
            # Fin d'Ajout GI 20250408

            table = soup.find("table", attrs={"summary": "Procedure Report: Rapport détaillé et/ou agrégé"})
            # Ajout GI 20250408
            if not table:
                raise ValueError(f"Table résultats non présente pour le finess: {finess}")
        
            headers = self._get_headers_from_table(table)

            data = []
            tbody = table.find("tbody")
        except ValueError as lib_erreur:
            click.echo(lib_erreur,nl=False)
            return []

        # remove last one which is a summary
        rows = tbody.find_all("tr")[:-1]
        for row in rows:
            cells = row.find_all("td")
            row_data = {headers[i]: cells[i].text.strip() for i in range(len(headers))}
            row_data = self._cast_row(row_data)
            data.append(row_data)
        # inject finess into each row
        data = [{"finess": finess} | _ for _ in data]

        return data

    def _cast_row(self, row: Dict[str, str]) -> Dict[str, str | int | float]:
        for col_name, casting_func in self.CASTMAPPING:
            row[col_name] = casting_func(row[col_name])
        return row

    def _get_headers_from_table(self, table: BeautifulSoup) -> List[str]:
        headers = []
        for header in table.find_all("th"):
            headers.append(header.text.strip())
        return headers


@click.command()
@click.option("--finess", help="list of finess codes (comma separated)")
@click.option("--finess_file", help="file_path to a file containing finess codes (one per line)", type=click.File("r"))
@click.option("--db_url", help="connexion url to MySQL database (mysql://user:pw@host:port/dbname) for inserting data")
@click.option("--file_ouput", help="file_path for collecting data", type=click.Path(exists=False))
@click.option("--db_init", help="(Re)create table into database", is_flag=True)
def main(finess, finess_file, db_url, file_ouput, db_init):
    # initialisation
    # Ajout GI
    # annee = 2023 
    annee = 2024
    # Fin Ajout GI
    click.secho("Initialisation", bold=True)

    # check if db_url or file_ouput is provided
    click.echo("Checking output")
    output_handler = None
    if db_url:
        click.echo(f"Connecting to database at {db_url}....", nl=False)
        output_handler = MysqlHandler(db_url)
        click.echo(click.style("OK", fg="green"))
    elif file_ouput:
        click.echo(f"Writing to file {file_ouput}")
        output_handler = FileHandler(file_ouput)
    if not output_handler:
        click.echo("Please provide a database URL or a file path for output")
        return
    click.secho("OK", fg="green", bold=True)

    if db_init:
        click.echo("Creating table....", nl=False)
        with output_handler:
            output_handler.create_table()
        click.secho("OK", fg="green", bold=True)

    # check if finess is provided
    click.echo("Checking input")
    finess_list = []
    if finess:
        finess_list += [_.strip() for _ in finess.split(",") if _.strip()]
    if finess_file:
        finess_list += [_.strip() for _ in finess_file.readlines() if _.strip()]
    if not finess_list:
        click.echo("Please provide a list of finess codes using --finess or --finess_file")
        return
    click.echo(f"{len(finess_list)} finess codes found")
    click.secho("OK", fg="green", bold=True)
    random.shuffle(finess_list) # Ajout GI 20250408

    # process
    click.secho("Start fetching data", fg="blue", bold=True)
    with output_handler:
        scraper = Scraper()
        for i, finess in enumerate(finess_list):
            # Debut Ajout GI 20250408: check that finess not already scrapped
            if output_handler.found_finess(finess, annee):
                continue
            # Fin Ajout GI 20250408    
            click.echo(f"Scraping {i} data for {finess}....", nl=False)
            data = scraper.get_data(finess, annee)
            if not data:
                # process next finess
                click.echo(f"No data found for {finess}")
                # Ajout GI 20250408
                # continue
            else:
                data = [{"annee": annee} | _ for _ in data]  
                # Fin d'ajout GI 20250408
                click.echo(f"fetched {len(data)} rows....", nl=False)
                for row in data:
                    output_handler.write(row)
                click.echo("wrote to output....", nl=False)
                click.secho("OK", fg="green", bold=True, nl=False)

            if i < len(finess_list) - 1:
                # do not wait when it is the last one
                for j in range(random.randint(15, 60)):
                    click.echo(".", nl=False)
                    sleep(30.0) # modif GI 20250419  (3.0)
                click.echo()

    click.echo(click.style("End fetching data", fg="blue", bold=True))


if __name__ == "__main__":
    main()
